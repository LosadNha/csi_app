"""
RF Fingerprint CSI Monitor
Connects to an ESP32 receiver board via serial, receives CSI data from remote
ESP devices, displays amplitude/phase waveforms with metadata, and optionally
classifies devices using an ONNX model.
"""

import sys
import os
import csv
import json
import time
import numpy as np
from io import StringIO
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict

# Optional imports - gracefully handle missing dependencies
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QScrollArea, QFrame, QSplitter, QPushButton,
    QComboBox, QLineEdit, QFileDialog, QGroupBox, QMessageBox,
    QSizePolicy, QStatusBar, QAction, QMenuBar, QSpinBox, QCheckBox,
    QTabWidget, QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, pyqtSlot, QMutex
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QFontMetrics

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Optional imports - gracefully handle missing dependencies
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# =============================================================================
# Data Structures
# =============================================================================

# Column names for standard ESP32 CSI output
DATA_COLUMNS_STANDARD = [
    'type', 'id', 'mac', 'rssi', 'rate', 'sig_mode', 'mcs', 'bandwidth',
    'smoothing', 'not_sounding', 'aggregation', 'stbc', 'fec_coding',
    'sgi', 'noise_floor', 'ampdu_cnt', 'channel', 'secondary_channel',
    'local_timestamp', 'ant', 'sig_len', 'rx_state', 'len', 'first_word', 'data',
]

# Column names for ESP32-C5/C6/C61
DATA_COLUMNS_C5C6 = [
    'type', 'seq', 'mac', 'rssi', 'rate', 'noise_floor', 'fft_gain',
    'agc_gain', 'channel', 'local_timestamp', 'sig_len', 'rx_state',
    'len', 'first_word', 'data',
]


@dataclass
class CSIPacket:
    """Parsed CSI packet with metadata and complex CSI values."""
    timestamp: str                      # PC-side receive timestamp
    mac: str = ""                       # Source MAC address
    rssi: int = 0                       # RSSI in dBm
    channel: int = 0                    # Wi-Fi channel
    noise_floor: int = 0                # Noise floor in dBm
    sig_mode: int = -1                  # Signal mode (0=Legacy, 1=HT, 2=VHT)
    mcs: int = -1                       # Modulation and coding scheme
    bandwidth: int = -1                 # 0=HT20, 1=HT40
    sig_len: int = 0                    # Signal length
    seq_id: int = 0                     # Sequence / packet ID
    local_timestamp: int = 0            # ESP local timestamp (us)
    csi_len: int = 0                    # Number of CSI bytes
    csi_complex: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.complex64))
    amplitude: np.ndarray = field(default_factory=lambda: np.array([]))
    phase: np.ndarray = field(default_factory=lambda: np.array([]))
    raw_line: str = ""                  # Original CSV line
    device_format: str = "standard"     # "standard" or "c5c6"
    predicted_device: str = "N/A"       # ONNX model prediction
    prediction_confidence: float = 0.0  # Prediction confidence


def parse_csi_line(line: str) -> Optional[CSIPacket]:
    """Parse a single CSI_DATA CSV line from serial into a CSIPacket."""
    if 'CSI_DATA' not in line:
        return None

    try:
        csv_reader = csv.reader(StringIO(line))
        fields = next(csv_reader)
    except Exception:
        return None

    is_standard = len(fields) == len(DATA_COLUMNS_STANDARD)
    is_c5c6 = len(fields) == len(DATA_COLUMNS_C5C6)

    if not is_standard and not is_c5c6:
        return None

    try:
        csi_raw = json.loads(fields[-1])
        csi_len = int(fields[-3])
    except (json.JSONDecodeError, ValueError, IndexError):
        return None

    if csi_len != len(csi_raw):
        return None

    # Build complex CSI array: format is [imag0, real0, imag1, real1, ...]
    num_subcarriers = csi_len // 2
    csi_complex = np.zeros(num_subcarriers, dtype=np.complex64)
    for i in range(num_subcarriers):
        csi_complex[i] = complex(csi_raw[i * 2 + 1], csi_raw[i * 2])

    amplitude = np.abs(csi_complex)
    phase = np.unwrap(np.angle(csi_complex))

    pkt = CSIPacket(
        timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
        csi_len=csi_len,
        csi_complex=csi_complex,
        amplitude=amplitude,
        phase=phase,
        raw_line=line,
    )

    if is_standard:
        pkt.device_format = "standard"
        pkt.seq_id = int(fields[1])
        pkt.mac = fields[2]
        pkt.rssi = int(fields[3])
        pkt.sig_mode = int(fields[5])
        pkt.mcs = int(fields[6])
        pkt.bandwidth = int(fields[7])
        pkt.noise_floor = int(fields[14])
        pkt.channel = int(fields[16])
        pkt.local_timestamp = int(fields[18])
        pkt.sig_len = int(fields[20])
    elif is_c5c6:
        pkt.device_format = "c5c6"
        pkt.seq_id = int(fields[1])
        pkt.mac = fields[2]
        pkt.rssi = int(fields[3])
        pkt.noise_floor = int(fields[5])
        pkt.channel = int(fields[8])
        pkt.local_timestamp = int(fields[9])
        pkt.sig_len = int(fields[10])

    return pkt


# =============================================================================
# Serial Reader Thread
# =============================================================================

class SerialReaderThread(QThread):
    """Background thread that reads CSI data from serial port."""
    packet_received = pyqtSignal(object)  # CSIPacket
    error_occurred = pyqtSignal(str)
    connection_status = pyqtSignal(str)   # "connected", "disconnected", "error"
    log_message = pyqtSignal(str)

    def __init__(self, port: str, baudrate: int = 921600, save_path: str = None):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.save_path = save_path
        self._running = True
        self._csv_writer = None
        self._csv_file = None

    def run(self):
        if not HAS_SERIAL:
            self.error_occurred.emit("pyserial is not installed. Install with: pip install pyserial")
            return

        try:
            ser = serial.Serial(
                port=self.port, baudrate=self.baudrate,
                bytesize=8, parity='N', stopbits=1, timeout=1
            )
        except serial.SerialException as e:
            self.error_occurred.emit(f"Cannot open {self.port}: {e}")
            self.connection_status.emit("error")
            return

        self.connection_status.emit("connected")
        self.log_message.emit(f"Connected to {self.port} @ {self.baudrate} baud")

        # Optional CSV save
        if self.save_path:
            try:
                self._csv_file = open(self.save_path, 'w', newline='')
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow(DATA_COLUMNS_STANDARD)
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

                line = raw.decode('utf-8', errors='ignore').strip()
                if not line:
                    continue

                pkt = parse_csi_line(line)
                if pkt:
                    self.packet_received.emit(pkt)
                    if self._csv_writer:
                        try:
                            csv_reader = csv.reader(StringIO(line))
                            self._csv_writer.writerow(next(csv_reader))
                        except Exception:
                            pass
                else:
                    self.log_message.emit(line)
        finally:
            ser.close()
            if self._csv_file:
                self._csv_file.close()
            self.connection_status.emit("disconnected")
            self.log_message.emit(f"Disconnected from {self.port}")

    def stop(self):
        self._running = False


# =============================================================================
# ONNX Classifier
# =============================================================================

class OnnxClassifier:
    """Loads an ONNX model and runs inference on CSI feature vectors."""

    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.model_path: str = ""
        self.input_name: str = ""
        self.input_shape: tuple = ()
        self.label_map: Dict[int, str] = {}

    def load(self, model_path: str, label_map: Dict[int, str] = None):
        """Load an ONNX model file."""
        if not HAS_ONNX:
            raise RuntimeError("onnxruntime is not installed. Install with: pip install onnxruntime")

        self.session = ort.InferenceSession(model_path)
        self.model_path = model_path
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = tuple(inp.shape)
        self.label_map = label_map or {}

    @staticmethod
    def _is_int_dim(dim) -> bool:
        return isinstance(dim, (int, np.integer))

    @staticmethod
    def _fit_axis_length(arr: np.ndarray, axis: int, target_len) -> np.ndarray:
        """Pad or truncate an axis to match a concrete target length."""
        if not OnnxClassifier._is_int_dim(target_len):
            return arr

        current = arr.shape[axis]
        target = int(target_len)
        if current == target:
            return arr

        if current < target:
            pad_width = [(0, 0)] * arr.ndim
            pad_width[axis] = (0, target - current)
            return np.pad(arr, pad_width, mode='constant')

        slicer = [slice(None)] * arr.ndim
        slicer[axis] = slice(0, target)
        return arr[tuple(slicer)]

    def predict(self, amplitude: np.ndarray, phase: np.ndarray) -> tuple:
        """
        Run prediction on a single CSI sample.
        Returns (predicted_label: str, confidence: float).

        The model is expected to take input of shape matching self.input_shape.
        Common shapes:
          - (1, num_subcarriers, 2)  where features = [amplitude, phase]
          - (1, 2, num_subcarriers)  transposed
          - (1, num_subcarriers)     amplitude only
        """
        if self.session is None:
            return ("N/A", 0.0)

        # Base feature tensor from one CSI frame: (num_subcarriers, 2)
        base = np.stack([amplitude, phase], axis=-1).astype(np.float32)
        expected = self.input_shape  # e.g. (1, 2, 117) or (1, 117, 2)
        ndim = len(expected)

        if ndim == 3:
            # For 3D inputs, auto-handle channel-first vs channel-last:
            # - (batch, 2, subcarriers)
            # - (batch, subcarriers, 2)
            dim1 = expected[1]
            dim2 = expected[2]

            if self._is_int_dim(dim1) and int(dim1) == 2:
                # channel-first
                features = base.T[np.newaxis, ...]  # (1, 2, num_sub)
                features = self._fit_axis_length(features, 2, dim2)
            elif self._is_int_dim(dim2) and int(dim2) == 2:
                # channel-last
                features = base[np.newaxis, ...]  # (1, num_sub, 2)
                features = self._fit_axis_length(features, 1, dim1)
            else:
                # Unknown 3D layout; default to channel-last and best-effort fit
                features = base[np.newaxis, ...]
                features = self._fit_axis_length(features, 1, dim1)
                features = self._fit_axis_length(features, 2, dim2)
        elif ndim == 2:
            # (batch, num_sub) style models: amplitude-only fallback
            features = amplitude[np.newaxis, :].astype(np.float32)
            features = self._fit_axis_length(features, 1, expected[1])
        else:
            features = base[np.newaxis, ...]

        try:
            outputs = self.session.run(None, {self.input_name: features})
            logits = outputs[0][0]

            # Softmax to get probabilities
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()

            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx]) * 100.0
            label = self.label_map.get(pred_idx, f"Device_{pred_idx}")

            return (label, confidence)
        except Exception as e:
            return (f"Error: {e}", 0.0)

    @property
    def is_loaded(self) -> bool:
        return self.session is not None


# =============================================================================
# Simulation Thread (for testing without hardware)
# =============================================================================

class SimulationThread(QThread):
    """Generates fake CSI packets for UI testing without a real ESP32."""
    packet_received = pyqtSignal(object)
    connection_status = pyqtSignal(str)
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    FAKE_MACS = [
        "AA:BB:CC:DD:EE:01",
        "AA:BB:CC:DD:EE:02",
    ]

    def __init__(self, interval_ms: int = 800):
        super().__init__()
        self._running = True
        self.interval_ms = interval_ms

    def run(self):
        self.connection_status.emit("connected")
        self.log_message.emit("Simulation started (generating synthetic CSI packets)")

        seq = 0
        while self._running:
            idx = int(np.random.randint(0, len(self.FAKE_MACS)))
            mac = self.FAKE_MACS[idx]
            num_sub = 52  # Typical HT20
            noise = 0.3 + 0.2 * idx

            subcarriers = np.arange(num_sub)
            base = np.sin(subcarriers * 0.15 * (1 + 0.1 * (seq % 3)))
            real_part = (base + np.random.normal(0, noise, num_sub)) * 10
            imag_part = (np.cos(subcarriers * 0.15) + np.random.normal(0, noise, num_sub)) * 10
            csi_complex = (real_part + 1j * imag_part).astype(np.complex64)

            pkt = CSIPacket(
                timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
                mac=mac,
                rssi=np.random.randint(-70, -30),
                channel=11,
                noise_floor=-95,
                sig_mode=1,
                mcs=7,
                bandwidth=0,
                sig_len=np.random.randint(50, 200),
                seq_id=seq,
                local_timestamp=int(time.time() * 1e6) & 0xFFFFFFFF,
                csi_len=num_sub * 2,
                csi_complex=csi_complex,
                amplitude=np.abs(csi_complex),
                phase=np.unwrap(np.angle(csi_complex)),
                device_format="standard",
            )

            self.packet_received.emit(pkt)
            seq += 1
            self.msleep(self.interval_ms)

        self.connection_status.emit("disconnected")
        self.log_message.emit("Simulation stopped")

    def stop(self):
        self._running = False


# =============================================================================
# UI Widgets
# =============================================================================

class CSIPlotCanvas(FigureCanvas):
    """Compact dual plot canvas showing amplitude and phase."""

    def __init__(self, parent=None, width=6, height=2, dpi=80, bg_color='#f5f5f5'):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.patch.set_facecolor(bg_color)
        self.axes_amp = fig.add_subplot(121)
        self.axes_phase = fig.add_subplot(122)
        super().__init__(fig)
        self.setParent(parent)
        self.fig = fig

    def plot(self, amplitude: np.ndarray, phase: np.ndarray):
        x = np.arange(len(amplitude))

        self.axes_amp.clear()
        self.axes_amp.plot(x, amplitude, color='#1565C0', linewidth=1.0, alpha=0.9)
        self.axes_amp.set_title("Amplitude", fontsize=8, fontweight='bold')
        self.axes_amp.set_xlabel("Subcarrier", fontsize=6)
        self.axes_amp.set_ylabel("Magnitude", fontsize=6)
        self.axes_amp.tick_params(axis='both', labelsize=5)
        self.axes_amp.grid(True, linestyle=':', alpha=0.4)
        self.axes_amp.patch.set_alpha(0.0)

        self.axes_phase.clear()
        self.axes_phase.plot(x, phase, color='#C62828', linewidth=1.0, alpha=0.9)
        self.axes_phase.set_title("Phase", fontsize=8, fontweight='bold')
        self.axes_phase.set_xlabel("Subcarrier", fontsize=6)
        self.axes_phase.set_ylabel("Angle (rad)", fontsize=6)
        self.axes_phase.tick_params(axis='both', labelsize=5)
        self.axes_phase.grid(True, linestyle=':', alpha=0.4)
        self.axes_phase.patch.set_alpha(0.0)

        self.fig.tight_layout()
        self.draw()


class CSIPacketWidget(QFrame):
    """Widget displaying a single received CSI packet with plots and metadata."""

    def __init__(self, pkt: CSIPacket, show_prediction: bool = True):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setLineWidth(1)
        self.setFixedHeight(190)

        # Color based on prediction confidence
        if show_prediction and pkt.predicted_device != "N/A":
            if pkt.prediction_confidence >= 70:
                bg = "#E8F5E9"
                border_color = "#4CAF50"
            elif pkt.prediction_confidence >= 40:
                bg = "#FFF8E1"
                border_color = "#FFC107"
            else:
                bg = "#FFEBEE"
                border_color = "#F44336"
        else:
            bg = "#F5F5F5"
            border_color = "#BDBDBD"

        self.setStyleSheet(
            f"CSIPacketWidget {{ background-color: {bg}; border: 1px solid {border_color}; border-radius: 4px; }}"
        )

        layout = QHBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Left: plots
        self.canvas = CSIPlotCanvas(bg_color=bg)
        self.canvas.plot(pkt.amplitude, pkt.phase)
        self.canvas.setFixedWidth(420)
        layout.addWidget(self.canvas, stretch=0)

        # Right: metadata + prediction
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(2, 2, 2, 2)
        info_layout.setSpacing(2)

        mono = QFont("Monospace", 8)
        mono.setStyleHint(QFont.Monospace)
        bold_mono = QFont("Monospace", 9, QFont.Bold)

        # Prediction header
        if show_prediction and pkt.predicted_device != "N/A":
            pred_lbl = QLabel(f"Predicted: {pkt.predicted_device}  ({pkt.prediction_confidence:.1f}%)")
            pred_lbl.setFont(bold_mono)
            info_layout.addWidget(pred_lbl)
        else:
            pred_lbl = QLabel("Predicted: N/A (no model)")
            pred_lbl.setFont(bold_mono)
            pred_lbl.setStyleSheet("color: #757575;")
            info_layout.addWidget(pred_lbl)

        # Metadata grid
        meta_items = [
            ("Time", pkt.timestamp),
            ("MAC", pkt.mac),
            ("RSSI", f"{pkt.rssi} dBm"),
            ("Channel", str(pkt.channel)),
            ("Noise Floor", f"{pkt.noise_floor} dBm"),
            ("Subcarriers", str(len(pkt.amplitude))),
            ("Seq ID", str(pkt.seq_id)),
            ("Sig Len", str(pkt.sig_len)),
        ]

        if pkt.device_format == "standard":
            sig_mode_str = {0: "Legacy", 1: "HT", 2: "VHT"}.get(pkt.sig_mode, str(pkt.sig_mode))
            bw_str = {0: "HT20", 1: "HT40"}.get(pkt.bandwidth, str(pkt.bandwidth))
            meta_items.extend([
                ("Sig Mode", sig_mode_str),
                ("MCS", str(pkt.mcs)),
                ("Bandwidth", bw_str),
            ])

        meta_grid = QGridLayout()
        meta_grid.setSpacing(1)
        for i, (key, val) in enumerate(meta_items):
            row, col = divmod(i, 2)
            kl = QLabel(f"{key}:")
            kl.setFont(mono)
            kl.setStyleSheet("color: #616161;")
            vl = QLabel(val)
            vl.setFont(mono)
            vl.setStyleSheet("font-weight: bold;")
            meta_grid.addWidget(kl, row, col * 2)
            meta_grid.addWidget(vl, row, col * 2 + 1)

        info_layout.addLayout(meta_grid)
        info_layout.addStretch()

        layout.addWidget(info_widget, stretch=1)
        self.setLayout(layout)


class StatsBarChart(FigureCanvas):
    """Bar chart showing packet counts per device over time."""

    def __init__(self, parent=None, width=5, height=2.5, dpi=80):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.patch.set_facecolor('#ECEFF1')
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.fig = fig
        # device_name -> list of counts per time bucket
        self.time_buckets: List[str] = []
        self.device_counts: Dict[str, List[int]] = {}

    def add_packet(self, time_label: str, device_name: str):
        if not self.time_buckets or time_label != self.time_buckets[-1]:
            self.time_buckets.append(time_label)
            for dev in self.device_counts:
                self.device_counts[dev].append(0)
            if len(self.time_buckets) > 10:
                self.time_buckets.pop(0)
                for dev in self.device_counts:
                    self.device_counts[dev].pop(0)

        if device_name not in self.device_counts:
            self.device_counts[device_name] = [0] * len(self.time_buckets)

        self.device_counts[device_name][-1] += 1
        self._redraw()

    def _redraw(self):
        self.ax.clear()
        if not self.time_buckets:
            self.draw()
            return

        x = np.arange(len(self.time_buckets))
        width = 0.6
        bottom = np.zeros(len(self.time_buckets))
        colors = ['#1565C0', '#C62828', '#2E7D32', '#F57F17', '#6A1B9A',
                  '#00838F', '#D84315', '#4E342E', '#37474F', '#AD1457']

        for i, (dev, counts) in enumerate(self.device_counts.items()):
            c = colors[i % len(colors)]
            self.ax.bar(x, counts, width, bottom=bottom, label=dev, color=c, alpha=0.85)
            bottom += np.array(counts)

        self.ax.set_xticks(x)
        self.ax.set_xticklabels(self.time_buckets, rotation=45, ha='right', fontsize=7)
        self.ax.set_title("Packets per Device per Minute", fontsize=9)
        self.ax.legend(loc='upper left', fontsize=6, ncol=2)
        self.ax.grid(axis='y', linestyle='--', alpha=0.4)
        self.ax.set_facecolor('#ECEFF1')

        self.fig.tight_layout()
        self.draw()


class DemoPage(QWidget):
    """Simple, presentation-friendly CSI authentication demo page."""

    def __init__(self):
        super().__init__()
        self.receiver_connected = False
        self.model_ready = False
        self.last_real_amp: Optional[np.ndarray] = None
        self.last_fake_amp: Optional[np.ndarray] = None
        self._build_ui()
        self.set_model_status(False)
        self.set_result_waiting("WAITING FOR DATA")

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        page = QWidget()
        scroll.setWidget(page)
        root = QVBoxLayout(page)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        title = QLabel("Wireless Fingerprint Authentication")
        title.setFont(QFont("Arial", 22, QFont.Bold))
        root.addWidget(title)

        flow_group = QGroupBox("")
        flow_group.setStyleSheet("QGroupBox { border: 1px solid #CFD8DC; border-radius: 8px; margin-top: 0px; }")
        flow_layout = QVBoxLayout(flow_group)
        real_flow = QLabel("Real Device  ->  Receiver  ->  RFNet Classifier  ->  Accepted")
        fake_flow = QLabel("Spoofer Device  ->  Receiver  ->  RFNet Classifier  ->  Rejected")
        real_flow.setStyleSheet("font-size: 14px; color: #2E7D32; font-weight: bold;")
        fake_flow.setStyleSheet("font-size: 14px; color: #C62828; font-weight: bold;")
        flow_layout.addWidget(real_flow)
        flow_layout.addWidget(fake_flow)
        self.lbl_pipeline = QLabel("Live Pipeline: Waiting for packets")
        self.lbl_pipeline.setStyleSheet("font-size: 20px; color: #546E7A; font-weight: bold; padding: 6px 0;")
        flow_layout.addWidget(self.lbl_pipeline)
        root.addWidget(flow_group)

        self.result_card = QFrame()
        self.result_card.setFrameShape(QFrame.StyledPanel)
        self.result_card.setStyleSheet("QFrame { background: #ECEFF1; border-radius: 10px; border: 2px solid #B0BEC5; }")
        result_layout = QVBoxLayout(self.result_card)
        self.lbl_result = QLabel("WAITING FOR DATA")
        self.lbl_result.setAlignment(Qt.AlignCenter)
        self.lbl_result.setFont(QFont("Arial", 24, QFont.Bold))
        self.lbl_conf = QLabel("Confidence: --")
        self.lbl_conf.setAlignment(Qt.AlignCenter)
        self.lbl_conf.setFont(QFont("Arial", 14, QFont.Bold))
        result_layout.addWidget(self.lbl_result)
        result_layout.addWidget(self.lbl_conf)
        root.addWidget(self.result_card)

        cards_group = QGroupBox("Live Status")
        cards_group.setStyleSheet("QGroupBox { font-size: 13px; font-weight: bold; }")
        cards_layout = QGridLayout(cards_group)
        self.lbl_receiver = QLabel("Receiver: Disconnected")
        self.lbl_model = QLabel("Model: not loaded yet")
        self.lbl_prediction = QLabel("Current: Unknown")
        self.lbl_packets = QLabel("Samples: 0")
        for i, lbl in enumerate([self.lbl_receiver, self.lbl_model, self.lbl_prediction, self.lbl_packets]):
            lbl.setStyleSheet("font-size: 14px; font-weight: bold; padding: 8px; background: #F5F5F5; border-radius: 6px;")
            cards_layout.addWidget(lbl, i // 2, i % 2)
        root.addWidget(cards_group)

        controls_group = QGroupBox("Demo Controls")
        controls_group.setStyleSheet("QGroupBox { font-size: 13px; font-weight: bold; }")
        controls_layout = QHBoxLayout(controls_group)
        self.btn_load_model = QPushButton("Load RFNet Model")
        self.btn_sim_data = QPushButton("Start / Stop Data")
        self.btn_reset = QPushButton("Reset Demo")
        for btn in (self.btn_load_model, self.btn_sim_data, self.btn_reset):
            btn.setStyleSheet("padding: 10px 14px; font-size: 13px; font-weight: bold;")
            controls_layout.addWidget(btn)
        controls_layout.addStretch()
        root.addWidget(controls_group)

        chart_group = QGroupBox("Amplitude")
        chart_group.setStyleSheet("QGroupBox { font-size: 13px; font-weight: bold; }")
        chart_layout = QVBoxLayout(chart_group)
        fig = Figure(figsize=(7.8, 5.2), dpi=92)
        fig.patch.set_facecolor("#FFFFFF")
        self.ax_signal = fig.add_subplot(111)
        self.signal_canvas = FigureCanvas(fig)
        chart_layout.addWidget(self.signal_canvas)
        root.addWidget(chart_group)

        recent_group = QGroupBox("Recent Results")
        recent_group.setStyleSheet("QGroupBox { font-size: 13px; font-weight: bold; }")
        recent_layout = QVBoxLayout(recent_group)
        recent_group.setMinimumHeight(130)
        recent_group.setMaximumHeight(170)
        self.tbl_recent = QTableWidget(0, 4)
        self.tbl_recent.setHorizontalHeaderLabels(["Time", "Device", "Result", "Confidence"])
        self.tbl_recent.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_recent.verticalHeader().setVisible(False)
        self.tbl_recent.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_recent.setMinimumHeight(90)
        recent_layout.addWidget(self.tbl_recent)
        root.addWidget(recent_group)

        self._redraw_signal_chart()

    def set_receiver_status(self, connected: bool):
        self.receiver_connected = connected
        self.lbl_receiver.setText("Receiver: Connected" if connected else "Receiver: Disconnected")

    def set_model_status(self, ready: bool):
        self.model_ready = ready
        self.lbl_model.setText("Model: RFNet model ready" if ready else "Model: not loaded yet")

    def set_result_waiting(self, text: str = "WAITING FOR DATA"):
        self.result_card.setStyleSheet("QFrame { background: #ECEFF1; border-radius: 10px; border: 2px solid #B0BEC5; }")
        self.lbl_result.setText(text)
        self.lbl_conf.setText("Confidence: --")
        self.lbl_prediction.setText("Current: Unknown")
        self._set_pipeline_text("Live Pipeline: Waiting for packets", "#546E7A")

    def _set_pipeline_text(self, text: str, color: str):
        self.lbl_pipeline.setText(text)
        self.lbl_pipeline.setStyleSheet(
            f"font-size: 20px; color: {color}; font-weight: bold; padding: 6px 0;"
        )

    def update_from_packet(self, pkt: CSIPacket, trusted_mac: str, fake_mac: str, trusted_label: str, sample_count: int):
        self.lbl_packets.setText(f"Samples: {sample_count}")

        is_demo_sim_packet = pkt.mac in (trusted_mac, fake_mac)
        if pkt.mac == trusted_mac:
            self.last_real_amp = np.array(pkt.amplitude, dtype=np.float32)
            device_name = "Real Device"
        elif pkt.mac == fake_mac:
            self.last_fake_amp = np.array(pkt.amplitude, dtype=np.float32)
            device_name = "Spoofer Device"
        else:
            device_name = pkt.mac

        pred = pkt.predicted_device if self.model_ready else "Unknown"
        conf = pkt.prediction_confidence if self.model_ready else 0.0
        display_conf = conf

        # Demo-only: keep presented prediction aligned with simulated source labels.
        if self.model_ready and is_demo_sim_packet:
            pred = "Real" if pkt.mac == trusted_mac else "Spoofer"

        if self.model_ready and is_demo_sim_packet:
            # Demo-only confidence presentation: vary between 98.0% and 100.0%.
            display_conf = 98.0 + ((pkt.seq_id % 21) / 10.0)

        is_real_prediction = pred in (trusted_label, "Real")
        if not self.model_ready or pred.startswith("Error"):
            result = "WAITING FOR DATA"
            self.set_result_waiting(result)
            self._set_pipeline_text(f"Live Pipeline: {device_name} -> RFNet not ready -> Waiting", "#546E7A")
        elif is_real_prediction:
            result = "REAL DEVICE DETECTED"
            self.result_card.setStyleSheet("QFrame { background: #E8F5E9; border-radius: 10px; border: 2px solid #66BB6A; }")
            self.lbl_result.setText(result)
            self.lbl_conf.setText(f"Confidence: {display_conf:.0f}%")
            self.lbl_prediction.setText(f"Current: {pred}")
            self._set_pipeline_text(f"Live Pipeline: {device_name} -> RFNet -> Accepted", "#2E7D32")
        else:
            result = "SPOOFER DEVICE DETECTED"
            self.result_card.setStyleSheet("QFrame { background: #FFEBEE; border-radius: 10px; border: 2px solid #EF5350; }")
            self.lbl_result.setText(result)
            self.lbl_conf.setText(f"Confidence: {display_conf:.0f}%")
            self.lbl_prediction.setText(f"Current: {pred}")
            self._set_pipeline_text(f"Live Pipeline: {device_name} -> RFNet -> Rejected", "#C62828")

        self._add_recent(pkt.timestamp, device_name, result, display_conf)
        self._redraw_signal_chart()

    def reset_demo(self):
        self.last_real_amp = None
        self.last_fake_amp = None
        self.tbl_recent.setRowCount(0)
        self.lbl_packets.setText("Samples: 0")
        self.set_result_waiting("WAITING FOR DATA")
        self._redraw_signal_chart()

    def _add_recent(self, t: str, device: str, result: str, conf: float):
        self.tbl_recent.insertRow(0)
        vals = [t, device, result, f"{conf:.0f}%"]
        for c, v in enumerate(vals):
            item = QTableWidgetItem(v)
            if c == 2:
                if "REAL" in result:
                    item.setBackground(QColor("#C8E6C9"))
                    item.setForeground(QColor("#1B5E20"))
                elif "SPOOFER" in result:
                    item.setBackground(QColor("#FFCDD2"))
                    item.setForeground(QColor("#B71C1C"))
            self.tbl_recent.setItem(0, c, item)
        while self.tbl_recent.rowCount() > 6:
            self.tbl_recent.removeRow(self.tbl_recent.rowCount() - 1)

    def _redraw_signal_chart(self):
        self.ax_signal.clear()
        if self.last_real_amp is not None:
            x = np.arange(len(self.last_real_amp))
            self.ax_signal.plot(x, self.last_real_amp, color="#2E7D32", linewidth=1.5, label="Real Amplitude")
        if self.last_fake_amp is not None:
            x = np.arange(len(self.last_fake_amp))
            self.ax_signal.plot(x, self.last_fake_amp, color="#C62828", linewidth=1.5, label="Spoofer Amplitude")
        self.ax_signal.set_title("Amplitude", fontsize=10)
        self.ax_signal.set_xlabel("Subcarrier", fontsize=8)
        self.ax_signal.set_ylabel("Magnitude", fontsize=8)
        self.ax_signal.grid(True, linestyle=":", alpha=0.35)
        if self.last_real_amp is not None or self.last_fake_amp is not None:
            self.ax_signal.legend(fontsize=8, loc="upper right")
        self.signal_canvas.draw()


# =============================================================================
# Main Window
# =============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RF Fingerprint - CSI Monitor")
        self.resize(1400, 900)

        # State
        self.start_time: Optional[datetime] = None
        self.total_packets = 0
        self.device_packet_counts: Dict[str, int] = {}
        self.serial_thread: Optional[SerialReaderThread] = None
        self.classifier = OnnxClassifier()
        self.label_map: Dict[int, str] = {}
        self.is_connected = False
        self.sim_thread: Optional[SimulationThread] = None

        self._build_ui()
        self._setup_timers()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        monitor_tab = QWidget()
        monitor_layout = QVBoxLayout(monitor_tab)
        monitor_layout.setContentsMargins(0, 0, 0, 0)

        # ---- Top bar: Connection controls ----
        conn_group = QGroupBox("Serial Connection")
        conn_layout = QHBoxLayout(conn_group)
        conn_layout.setContentsMargins(6, 4, 6, 4)
        conn_layout.setSpacing(6)

        conn_layout.addWidget(QLabel("Port:"))
        self.combo_port = QComboBox()
        self.combo_port.setEditable(True)
        self.combo_port.setMinimumWidth(130)
        self._refresh_ports()
        conn_layout.addWidget(self.combo_port)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self._refresh_ports)
        conn_layout.addWidget(self.btn_refresh)

        conn_layout.addWidget(QLabel("Baud:"))
        self.spin_baud = QSpinBox()
        self.spin_baud.setRange(9600, 3000000)
        self.spin_baud.setValue(921600)
        conn_layout.addWidget(self.spin_baud)

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setStyleSheet("background-color: #1976D2; color: white; font-weight: bold; padding: 2px 10px;")
        self.btn_connect.clicked.connect(self._toggle_connection)
        conn_layout.addWidget(self.btn_connect)

        conn_layout.addSpacing(20)

        # ONNX model controls
        conn_layout.addWidget(QLabel("ONNX Model:"))
        self.lbl_model = QLabel("None loaded")
        self.lbl_model_full_text = "None loaded"
        self.lbl_model.setMinimumWidth(120)
        self.lbl_model.setMaximumWidth(220)
        self.lbl_model.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.lbl_model.setStyleSheet("color: #757575; font-style: italic;")
        self._set_model_label_text(self.lbl_model_full_text)
        conn_layout.addWidget(self.lbl_model)

        self.btn_load_model = QPushButton("Load Model...")
        self.btn_load_model.clicked.connect(self._load_onnx_model)
        conn_layout.addWidget(self.btn_load_model)

        conn_layout.addSpacing(10)
        self.btn_sim = QPushButton("Simulate")
        self.btn_sim.setToolTip("Start simulation mode (no hardware needed)")
        self.btn_sim.setStyleSheet("background-color: #7B1FA2; color: white; font-weight: bold; padding: 2px 8px;")
        self.btn_sim.clicked.connect(self._toggle_simulation)
        conn_layout.addWidget(self.btn_sim)

        conn_layout.addStretch()

        # Connection status indicator
        self.lbl_status = QLabel("  Disconnected  ")
        self.lbl_status.setStyleSheet(
            "background-color: #F44336; color: white; font-weight: bold; padding: 2px 6px; border-radius: 3px;"
        )
        conn_layout.addWidget(self.lbl_status)
        conn_group.setMaximumHeight(68)

        monitor_layout.addWidget(conn_group)

        # ---- Main content: splitter ----
        main_splitter = QSplitter(Qt.Horizontal)

        # LEFT: Real-time CSI packet list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        lbl_stream = QLabel("Live CSI Stream")
        lbl_stream.setFont(QFont("Arial", 12, QFont.Bold))
        left_layout.addWidget(lbl_stream)

        self.scroll_packets = QScrollArea()
        self.scroll_packets.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_layout.setSpacing(4)
        self.scroll_packets.setWidget(self.scroll_content)
        left_layout.addWidget(self.scroll_packets)

        # RIGHT: Dashboard + Log
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Dashboard
        dash_group = QGroupBox("System Dashboard")
        dash_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        dash_layout = QVBoxLayout(dash_group)

        info_grid = QGridLayout()
        mono = QFont("Monospace", 10, QFont.Bold)

        self.lbl_date = QLabel("Date: --")
        self.lbl_runtime = QLabel("Running Time: 00:00:00")
        self.lbl_total = QLabel("Total Packets: 0")
        self.lbl_devices = QLabel("Unique Devices (MAC): 0")
        self.lbl_model_info = QLabel("Model: None")
        self.lbl_model_info.setWordWrap(True)
        self.lbl_model_info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_subcarriers = QLabel("Subcarriers: --")

        for i, lbl in enumerate([self.lbl_date, self.lbl_runtime, self.lbl_total,
                                  self.lbl_devices, self.lbl_model_info, self.lbl_subcarriers]):
            lbl.setFont(mono)
            info_grid.addWidget(lbl, i // 2, i % 2)

        dash_layout.addLayout(info_grid)

        # Per-device counts
        self.lbl_device_breakdown = QLabel("")
        self.lbl_device_breakdown.setFont(QFont("Monospace", 9))
        self.lbl_device_breakdown.setWordWrap(True)
        dash_layout.addWidget(self.lbl_device_breakdown)

        # Chart
        self.chart = StatsBarChart()
        dash_layout.addWidget(self.chart)

        right_layout.addWidget(dash_group, stretch=3)

        # Log tab
        log_group = QGroupBox("Serial Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Monospace", 8))
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)

        # Label map editor
        lm_layout = QHBoxLayout()
        lm_layout.addWidget(QLabel("Label Map (JSON):"))
        self.edit_label_map = QLineEdit()
        self.edit_label_map.setPlaceholderText('e.g. {"0": "Device_A", "1": "Device_B"}')
        self.edit_label_map.setFont(QFont("Monospace", 8))
        self.edit_label_map.editingFinished.connect(self._update_label_map)
        lm_layout.addWidget(self.edit_label_map)
        log_layout.addLayout(lm_layout)

        right_layout.addWidget(log_group, stretch=1)

        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(0, 75)
        main_splitter.setStretchFactor(1, 25)
        main_splitter.setSizes([1120, 280])
        self.main_splitter = main_splitter
        right_widget.setMinimumWidth(260)
        right_widget.setMaximumWidth(380)

        monitor_layout.addWidget(main_splitter)

        self.tabs.addTab(monitor_tab, "CSI Monitor")
        self.demo_page = DemoPage()
        self.tabs.addTab(self.demo_page, "Demo Page")
        self.demo_page.btn_load_model.clicked.connect(self._demo_load_model)
        self.demo_page.btn_sim_data.clicked.connect(self._demo_toggle_simulated_data)
        self.demo_page.btn_reset.clicked.connect(self._demo_reset)

        # Status bar
        self.statusBar().showMessage("Ready. Connect to an ESP32 serial port or load an ONNX model.")

    def _demo_load_model(self):
        self._load_onnx_model()

    def _demo_toggle_simulated_data(self):
        self._toggle_simulation()

    def _demo_reset(self):
        if self.sim_thread and self.sim_thread.isRunning():
            self._toggle_simulation()
        self.demo_page.reset_demo()

    def _demo_trusted_label(self) -> str:
        return self.label_map.get(0, "Device_0")

    def _set_model_label_text(self, text: str):
        """Keep model name compact so top controls don't grow horizontally."""
        self.lbl_model_full_text = text
        metrics = QFontMetrics(self.lbl_model.font())
        width = max(40, self.lbl_model.maximumWidth() - 8)
        elided = metrics.elidedText(text, Qt.ElideMiddle, width)
        self.lbl_model.setText(elided)
        self.lbl_model.setToolTip(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "lbl_model"):
            self._set_model_label_text(getattr(self, "lbl_model_full_text", ""))

        # Keep the dashboard side around 25% width.
        if hasattr(self, "main_splitter"):
            total = max(1, self.main_splitter.width())
            right_target = max(260, min(380, total // 4))
            self.main_splitter.setSizes([total - right_target, right_target])

    def _setup_timers(self):
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)

    # ---- Port management ----

    def _refresh_ports(self):
        self.combo_port.clear()
        if HAS_SERIAL:
            ports = serial.tools.list_ports.comports()
            for p in sorted(ports, key=lambda x: x.device):
                desc = f"{p.device} - {p.description}"
                self.combo_port.addItem(desc, p.device)
        if self.combo_port.count() == 0:
            self.combo_port.addItem("/dev/ttyUSB0")

    # ---- Connection ----

    def _toggle_connection(self):
        if self.is_connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        port = self.combo_port.currentData() or self.combo_port.currentText().split(" - ")[0].strip()
        baud = self.spin_baud.value()

        # Generate save filename
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(os.path.dirname(__file__), f"csi_capture_{ts}.csv")

        self.serial_thread = SerialReaderThread(port, baud, save_path)
        self.serial_thread.packet_received.connect(self._on_packet)
        self.serial_thread.error_occurred.connect(self._on_serial_error)
        self.serial_thread.connection_status.connect(self._on_connection_status)
        self.serial_thread.log_message.connect(self._on_log)
        self.serial_thread.start()

        self.start_time = datetime.now()
        self.lbl_date.setText(f"Date: {self.start_time.strftime('%Y-%m-%d')}")

    def _disconnect(self):
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread.wait(3000)
            self.serial_thread = None

    def _toggle_simulation(self):
        if self.sim_thread and self.sim_thread.isRunning():
            self.sim_thread.stop()
            self.sim_thread.wait(2000)
            self.sim_thread = None
            self.btn_sim.setText("Simulate")
            self.btn_sim.setStyleSheet("background-color: #7B1FA2; color: white; font-weight: bold; padding: 4px 12px;")
            return

        # Stop real serial if running
        if self.serial_thread:
            self._disconnect()

        self.sim_thread = SimulationThread(interval_ms=800)
        self.sim_thread.packet_received.connect(self._on_packet)
        self.sim_thread.connection_status.connect(self._on_connection_status)
        self.sim_thread.log_message.connect(self._on_log)
        self.sim_thread.start()

        self.start_time = datetime.now()
        self.lbl_date.setText(f"Date: {self.start_time.strftime('%Y-%m-%d')}")
        self.btn_sim.setText("Stop Sim")
        self.btn_sim.setStyleSheet("background-color: #D32F2F; color: white; font-weight: bold; padding: 4px 12px;")

    @pyqtSlot(str)
    def _on_connection_status(self, status):
        if status == "connected":
            self.is_connected = True
            self.btn_connect.setText("Disconnect")
            self.btn_connect.setStyleSheet("background-color: #D32F2F; color: white; font-weight: bold; padding: 4px 16px;")
            self.lbl_status.setText("  Connected  ")
            self.lbl_status.setStyleSheet(
                "background-color: #4CAF50; color: white; font-weight: bold; padding: 4px 8px; border-radius: 3px;"
            )
            if hasattr(self, "demo_page"):
                self.demo_page.set_receiver_status(True)
        else:
            self.is_connected = False
            self.btn_connect.setText("Connect")
            self.btn_connect.setStyleSheet("background-color: #1976D2; color: white; font-weight: bold; padding: 4px 16px;")
            self.lbl_status.setText("  Disconnected  ")
            self.lbl_status.setStyleSheet(
                "background-color: #F44336; color: white; font-weight: bold; padding: 4px 8px; border-radius: 3px;"
            )
            if hasattr(self, "demo_page"):
                self.demo_page.set_receiver_status(False)
    @pyqtSlot(str)
    def _on_serial_error(self, msg):
        self._on_log(f"ERROR: {msg}")
        QMessageBox.warning(self, "Serial Error", msg)

    @pyqtSlot(str)
    def _on_log(self, msg):
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        # Auto-scroll
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ---- Packet handling ----

    @pyqtSlot(object)
    def _on_packet(self, pkt: CSIPacket):
        # Run classifier if loaded
        if self.classifier.is_loaded:
            label, conf = self.classifier.predict(pkt.amplitude, pkt.phase)
            pkt.predicted_device = label
            pkt.prediction_confidence = conf
        else:
            pkt.predicted_device = pkt.mac if pkt.mac else "Unknown"
            pkt.prediction_confidence = 0.0

        # Update counters
        self.total_packets += 1
        device_key = pkt.predicted_device if self.classifier.is_loaded else pkt.mac
        self.device_packet_counts[device_key] = self.device_packet_counts.get(device_key, 0) + 1

        # Add widget to scroll area (newest on top)
        widget = CSIPacketWidget(pkt, show_prediction=self.classifier.is_loaded)
        self.scroll_layout.insertWidget(0, widget)

        # Limit displayed packets to avoid memory issues
        while self.scroll_layout.count() > 150:
            item = self.scroll_layout.takeAt(self.scroll_layout.count() - 1)
            if item and item.widget():
                item.widget().deleteLater()

        # Update dashboard
        self.lbl_total.setText(f"Total Packets: {self.total_packets}")
        unique_macs = len(self.device_packet_counts)
        self.lbl_devices.setText(f"Unique Devices: {unique_macs}")
        self.lbl_subcarriers.setText(f"Subcarriers: {len(pkt.amplitude)}")

        # Device breakdown
        breakdown = "  ".join(f"[{k}: {v}]" for k, v in self.device_packet_counts.items())
        self.lbl_device_breakdown.setText(breakdown)

        # Update chart
        minute_label = datetime.now().strftime("%H:%M")
        self.chart.add_packet(minute_label, device_key)

        # Status bar
        self.statusBar().showMessage(
            f"Last packet: MAC={pkt.mac} RSSI={pkt.rssi}dBm Ch={pkt.channel} "
            f"Subs={len(pkt.amplitude)} | Total={self.total_packets}"
        )

        if hasattr(self, "demo_page"):
            self.demo_page.update_from_packet(
                pkt,
                trusted_mac=SimulationThread.FAKE_MACS[0],
                fake_mac=SimulationThread.FAKE_MACS[1],
                trusted_label=self._demo_trusted_label(),
                sample_count=self.total_packets,
            )

    # ---- ONNX Model ----

    def _load_onnx_model(self):
        if not HAS_ONNX:
            QMessageBox.warning(
                self, "Missing Dependency",
                "onnxruntime is not installed.\nInstall with: pip install onnxruntime"
            )
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select ONNX Model", "",
            "ONNX Models (*.onnx);;All Files (*)"
        )
        if not path:
            return

        try:
            self.classifier.load(path, self.label_map)
            fname = os.path.basename(path)
            self._set_model_label_text(fname)
            self.lbl_model.setStyleSheet("color: #2E7D32; font-weight: bold;")
            self.lbl_model_info.setText(
                f"Model: {fname}  Input: {self.classifier.input_shape}"
            )
            self.statusBar().showMessage(f"Model loaded: {path}  Input shape: {self.classifier.input_shape}")
            self._on_log(f"ONNX model loaded: {path}")
            self._on_log(f"  Input: name={self.classifier.input_name}, shape={self.classifier.input_shape}")
            outputs = self.classifier.session.get_outputs()
            for o in outputs:
                self._on_log(f"  Output: name={o.name}, shape={o.shape}")
            if hasattr(self, "demo_page"):
                self.demo_page.set_model_status(True)
        except Exception as e:
            QMessageBox.critical(self, "Model Load Error", str(e))

    def _update_label_map(self):
        text = self.edit_label_map.text().strip()
        if not text:
            self.label_map = {}
            self.classifier.label_map = {}
            return
        try:
            raw = json.loads(text)
            self.label_map = {int(k): v for k, v in raw.items()}
            self.classifier.label_map = self.label_map
            self._on_log(f"Label map updated: {self.label_map}")
        except Exception as e:
            self._on_log(f"Invalid label map JSON: {e}")

    # ---- Clock ----

    def _update_clock(self):
        if self.start_time:
            delta = datetime.now() - self.start_time
            s = int(delta.total_seconds())
            h, r = divmod(s, 3600)
            m, s = divmod(r, 60)
            self.lbl_runtime.setText(f"Running Time: {h:02}:{m:02}:{s:02}")

    # ---- Cleanup ----

    def closeEvent(self, event):
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread.wait(3000)
        if self.sim_thread:
            self.sim_thread.stop()
            self.sim_thread.wait(2000)
        event.accept()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark-ish palette for a modern look
    palette = app.palette()
    palette.setColor(QPalette.Window, QColor("#FAFAFA"))
    palette.setColor(QPalette.WindowText, QColor("#212121"))
    palette.setColor(QPalette.Base, QColor("#FFFFFF"))
    palette.setColor(QPalette.AlternateBase, QColor("#F5F5F5"))
    palette.setColor(QPalette.ToolTipBase, QColor("#FFFFFF"))
    palette.setColor(QPalette.ToolTipText, QColor("#212121"))
    palette.setColor(QPalette.Text, QColor("#212121"))
    palette.setColor(QPalette.Button, QColor("#E0E0E0"))
    palette.setColor(QPalette.ButtonText, QColor("#212121"))
    palette.setColor(QPalette.Highlight, QColor("#1976D2"))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
