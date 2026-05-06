"""Background thread that reads CSI frames from a serial port."""

import csv
from io import StringIO

from PyQt5.QtCore import QThread, pyqtSignal

from .models import DATA_COLUMNS_STANDARD
from .parser import parse_csi_line

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


class SerialReaderThread(QThread):
    """Read lines from an ESP32 serial port, parse CSI_DATA, emit packets."""

    packet_received = pyqtSignal(object)   # CSIPacket
    error_occurred = pyqtSignal(str)
    connection_status = pyqtSignal(str)    # "connected" | "disconnected" | "error"
    log_message = pyqtSignal(str)

    def __init__(self, port: str, baudrate: int = 921600, save_path: str | None = None):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.save_path = save_path
        self._running = True

    def run(self):
        if not HAS_SERIAL:
            self.error_occurred.emit("pyserial not installed. pip install pyserial")
            return

        try:
            ser = serial.Serial(
                port=self.port, baudrate=self.baudrate,
                bytesize=8, parity="N", stopbits=1, timeout=1,
            )
        except serial.SerialException as e:
            self.error_occurred.emit(f"Cannot open {self.port}: {e}")
            self.connection_status.emit("error")
            return

        self.connection_status.emit("connected")
        self.log_message.emit(f"Connected to {self.port} @ {self.baudrate} baud")

        csv_file = csv_writer = None
        if self.save_path:
            try:
                csv_file = open(self.save_path, "w", newline="")
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(DATA_COLUMNS_STANDARD)
            except Exception as e:
                self.log_message.emit(f"Warning: cannot open save file: {e}")

        try:
            while self._running:
                try:
                    raw = ser.readline()
                except serial.SerialException as e:
                    self.error_occurred.emit(f"Serial read error: {e}")
                    break

                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                pkt = parse_csi_line(line)
                if pkt:
                    self.packet_received.emit(pkt)
                    if csv_writer:
                        try:
                            csv_writer.writerow(next(csv.reader(StringIO(line))))
                        except Exception:
                            pass
                else:
                    self.log_message.emit(line)
        finally:
            ser.close()
            if csv_file:
                csv_file.close()
            self.connection_status.emit("disconnected")
            self.log_message.emit(f"Disconnected from {self.port}")

    def stop(self):
        self._running = False

    @staticmethod
    def available_ports() -> list:
        """Return list of (device, description) tuples."""
        if not HAS_SERIAL:
            return []
        return [
            (p.device, p.description)
            for p in sorted(serial.tools.list_ports.comports(), key=lambda x: x.device)
        ]
