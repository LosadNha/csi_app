"""MainWindow: thin orchestrator wiring core backends to UI panels."""

import os
import json
from datetime import datetime
from typing import Optional, Dict

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QSplitter, QFileDialog, QMessageBox,
)
from PyQt5.QtCore import QTimer, Qt, pyqtSlot

from core.models import CSIPacket
from core.classifier import OnnxClassifier, HAS_ONNX
from core.serial_reader import SerialReaderThread
from core.simulator import SimulationThread

from .connection_panel import ConnectionPanel
from .packet_view import PacketStreamPanel
from .dashboard_panel import DashboardPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RF Fingerprint - CSI Monitor")
        self.resize(1400, 900)

        # ── Backend state ─────────────────────────────────────────────────
        self._start_time: Optional[datetime] = None
        self._total_packets = 0
        self._device_counts: Dict[str, int] = {}
        self._serial_thread: Optional[SerialReaderThread] = None
        self._sim_thread: Optional[SimulationThread] = None
        self._classifier = OnnxClassifier()
        self._label_map: Dict[int, str] = {}

        # ── Build UI ──────────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        # Top bar
        self._conn_panel = ConnectionPanel()
        self._conn_panel.populate_ports()
        root.addWidget(self._conn_panel)

        # Content splitter
        splitter = QSplitter(Qt.Horizontal)

        self._stream = PacketStreamPanel()
        self._dashboard = DashboardPanel()

        splitter.addWidget(self._stream)
        splitter.addWidget(self._dashboard)
        splitter.setStretchFactor(0, 60)
        splitter.setStretchFactor(1, 40)
        root.addWidget(splitter)

        self.statusBar().showMessage("Ready. Connect to an ESP32 or start simulation.")

        # ── Wire signals ──────────────────────────────────────────────────
        self._conn_panel.connect_requested.connect(self._on_connect)
        self._conn_panel.disconnect_requested.connect(self._on_disconnect)
        self._conn_panel.refresh_requested.connect(self._conn_panel.populate_ports)
        self._conn_panel.simulate_requested.connect(self._on_start_sim)
        self._conn_panel.stop_sim_requested.connect(self._on_stop_sim)
        self._conn_panel.load_model_requested.connect(self._on_load_model)
        self._dashboard.label_map_changed.connect(self._on_label_map_text)

        # Clock
        self._clock = QTimer()
        self._clock.timeout.connect(self._tick_clock)
        self._clock.start(1000)

    # =====================================================================
    # Serial connection
    # =====================================================================

    @pyqtSlot(str, int)
    def _on_connect(self, port: str, baud: int):
        self._stop_sim_thread()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"csi_capture_{ts}.csv")

        t = SerialReaderThread(port, baud, save)
        t.packet_received.connect(self._on_packet)
        t.error_occurred.connect(self._on_error)
        t.connection_status.connect(self._on_conn_status)
        t.log_message.connect(self._dashboard.append_log)
        t.start()
        self._serial_thread = t

        self._mark_start()

    @pyqtSlot()
    def _on_disconnect(self):
        self._stop_serial_thread()

    def _stop_serial_thread(self):
        if self._serial_thread:
            self._serial_thread.stop()
            self._serial_thread.wait(3000)
            self._serial_thread = None

    # =====================================================================
    # Simulation
    # =====================================================================

    @pyqtSlot()
    def _on_start_sim(self):
        self._stop_serial_thread()

        t = SimulationThread(interval_ms=800)
        t.packet_received.connect(self._on_packet)
        t.connection_status.connect(self._on_conn_status)
        t.log_message.connect(self._dashboard.append_log)
        t.start()
        self._sim_thread = t

        self._conn_panel.set_simulating(True)
        self._mark_start()

    @pyqtSlot()
    def _on_stop_sim(self):
        self._stop_sim_thread()
        self._conn_panel.set_simulating(False)

    def _stop_sim_thread(self):
        if self._sim_thread:
            self._sim_thread.stop()
            self._sim_thread.wait(2000)
            self._sim_thread = None

    # =====================================================================
    # Connection status
    # =====================================================================

    @pyqtSlot(str)
    def _on_conn_status(self, status: str):
        connected = status == "connected"
        self._conn_panel.set_connected(connected)

    @pyqtSlot(str)
    def _on_error(self, msg: str):
        self._dashboard.append_log(f"ERROR: {msg}")
        QMessageBox.warning(self, "Serial Error", msg)

    # =====================================================================
    # Packet handling (the central data path)
    # =====================================================================

    @pyqtSlot(object)
    def _on_packet(self, pkt: CSIPacket):
        # Classify
        if self._classifier.is_loaded:
            label, conf = self._classifier.predict(pkt.amplitude, pkt.phase)
            pkt.predicted_device = label
            pkt.prediction_confidence = conf
        else:
            pkt.predicted_device = pkt.mac or "Unknown"
            pkt.prediction_confidence = 0.0

        # Counters
        self._total_packets += 1
        key = pkt.predicted_device if self._classifier.is_loaded else pkt.mac
        self._device_counts[key] = self._device_counts.get(key, 0) + 1

        # Update UI
        self._stream.add_packet(pkt, show_prediction=self._classifier.is_loaded)

        self._dashboard.set_total_packets(self._total_packets)
        self._dashboard.set_unique_devices(len(self._device_counts))
        self._dashboard.set_subcarriers(pkt.num_subcarriers)
        self._dashboard.set_device_breakdown(self._device_counts)
        self._dashboard.add_chart_point(datetime.now().strftime("%H:%M"), key)

        self.statusBar().showMessage(
            f"Last: MAC={pkt.mac}  RSSI={pkt.rssi}dBm  Ch={pkt.channel}  "
            f"Subs={pkt.num_subcarriers} | Total={self._total_packets}"
        )

    # =====================================================================
    # ONNX model
    # =====================================================================

    @pyqtSlot()
    def _on_load_model(self):
        if not HAS_ONNX:
            QMessageBox.warning(
                self, "Missing Dependency",
                "onnxruntime not installed.\npip install onnxruntime",
            )
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select ONNX Model", "", "ONNX Models (*.onnx);;All Files (*)"
        )
        if not path:
            return

        try:
            self._classifier.load(path, self._label_map)
            name = os.path.basename(path)
            self._conn_panel.set_model_name(name)
            self._dashboard.set_model_info(f"{name}  Input: {self._classifier.input_shape}")
            self._dashboard.append_log(f"ONNX model loaded: {path}")
            self._dashboard.append_log(
                f"  Input: name={self._classifier.input_name}, shape={self._classifier.input_shape}"
            )
            for o in self._classifier.session.get_outputs():
                self._dashboard.append_log(f"  Output: name={o.name}, shape={o.shape}")
        except Exception as e:
            QMessageBox.critical(self, "Model Load Error", str(e))

    @pyqtSlot(str)
    def _on_label_map_text(self, text: str):
        text = text.strip()
        if not text:
            self._label_map = {}
            self._classifier.label_map = {}
            return
        try:
            raw = json.loads(text)
            self._label_map = {int(k): v for k, v in raw.items()}
            self._classifier.label_map = self._label_map
            self._dashboard.append_log(f"Label map updated: {self._label_map}")
        except Exception as e:
            self._dashboard.append_log(f"Invalid label map JSON: {e}")

    # =====================================================================
    # Clock / lifecycle
    # =====================================================================

    def _mark_start(self):
        self._start_time = datetime.now()
        self._dashboard.set_date(self._start_time.strftime("%Y-%m-%d"))

    def _tick_clock(self):
        if self._start_time:
            s = int((datetime.now() - self._start_time).total_seconds())
            h, r = divmod(s, 3600)
            m, s = divmod(r, 60)
            self._dashboard.set_runtime(h, m, s)

    def closeEvent(self, event):
        self._stop_serial_thread()
        self._stop_sim_thread()
        event.accept()
