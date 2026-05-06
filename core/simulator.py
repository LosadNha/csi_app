"""Generate synthetic CSI packets for UI testing without hardware."""

import time
import numpy as np
from datetime import datetime

from PyQt5.QtCore import QThread, pyqtSignal

from .models import CSIPacket

FAKE_MACS = [
    "AA:BB:CC:DD:EE:01",
    "AA:BB:CC:DD:EE:02",
    "AA:BB:CC:DD:EE:03",
]


class SimulationThread(QThread):
    """Emit one synthetic CSIPacket per interval."""

    packet_received = pyqtSignal(object)
    connection_status = pyqtSignal(str)
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, interval_ms: int = 800):
        super().__init__()
        self._running = True
        self.interval_ms = interval_ms

    def run(self):
        self.connection_status.emit("connected")
        self.log_message.emit("Simulation started (generating synthetic CSI packets)")

        seq = 0
        while self._running:
            mac = FAKE_MACS[seq % len(FAKE_MACS)]
            n = 52  # HT20
            noise = 0.3 + 0.2 * (seq % len(FAKE_MACS))

            x = np.arange(n)
            r = (np.sin(x * 0.15 * (1 + 0.1 * (seq % 3))) + np.random.normal(0, noise, n)) * 10
            im = (np.cos(x * 0.15) + np.random.normal(0, noise, n)) * 10
            csi = (r + 1j * im).astype(np.complex64)

            pkt = CSIPacket(
                timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
                mac=mac,
                rssi=int(np.random.randint(-70, -30)),
                channel=11,
                noise_floor=-95,
                sig_mode=1,
                mcs=7,
                bandwidth=0,
                sig_len=int(np.random.randint(50, 200)),
                seq_id=seq,
                local_timestamp=int(time.time() * 1e6) & 0xFFFFFFFF,
                csi_len=n * 2,
                csi_complex=csi,
                amplitude=np.abs(csi),
                phase=np.unwrap(np.angle(csi)),
                device_format="standard",
            )

            self.packet_received.emit(pkt)
            seq += 1
            self.msleep(self.interval_ms)

        self.connection_status.emit("disconnected")
        self.log_message.emit("Simulation stopped")

    def stop(self):
        self._running = False
