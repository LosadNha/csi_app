"""Top toolbar: serial port selector, baud rate, connect/simulate, ONNX model loader."""

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QGroupBox, QLabel, QComboBox,
    QPushButton, QSpinBox, QLineEdit,
)
from PyQt5.QtCore import pyqtSignal

from core.serial_reader import SerialReaderThread
from .style import Colors


class ConnectionPanel(QGroupBox):
    """Emits high-level intents; actual connection logic lives in MainWindow."""

    connect_requested = pyqtSignal(str, int)        # port, baudrate
    disconnect_requested = pyqtSignal()
    refresh_requested = pyqtSignal()
    simulate_requested = pyqtSignal()
    stop_sim_requested = pyqtSignal()
    load_model_requested = pyqtSignal()
    label_map_changed = pyqtSignal(str)             # raw JSON text

    def __init__(self, parent=None):
        super().__init__("Serial Connection", parent)
        self._connected = False
        self._simulating = False
        self._build()

    def _build(self):
        layout = QHBoxLayout(self)

        # Port
        layout.addWidget(QLabel("Port:"))
        self.combo_port = QComboBox()
        self.combo_port.setEditable(True)
        self.combo_port.setMinimumWidth(160)
        layout.addWidget(self.combo_port)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh_requested.emit)
        layout.addWidget(self.btn_refresh)

        # Baud
        layout.addWidget(QLabel("Baud:"))
        self.spin_baud = QSpinBox()
        self.spin_baud.setRange(9600, 3_000_000)
        self.spin_baud.setValue(921600)
        layout.addWidget(self.spin_baud)

        # Connect
        self.btn_connect = QPushButton("Connect")
        self._apply_connect_style(False)
        self.btn_connect.clicked.connect(self._on_connect_clicked)
        layout.addWidget(self.btn_connect)

        layout.addSpacing(20)

        # ONNX model
        layout.addWidget(QLabel("ONNX Model:"))
        self.lbl_model = QLabel("None loaded")
        self.lbl_model.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-style: italic;")
        layout.addWidget(self.lbl_model)

        self.btn_load_model = QPushButton("Load Model...")
        self.btn_load_model.clicked.connect(self.load_model_requested.emit)
        layout.addWidget(self.btn_load_model)

        layout.addSpacing(10)

        # Simulate
        self.btn_sim = QPushButton("Simulate")
        self.btn_sim.setToolTip("Generate synthetic packets (no hardware)")
        self._apply_sim_style(False)
        self.btn_sim.clicked.connect(self._on_sim_clicked)
        layout.addWidget(self.btn_sim)

        layout.addStretch()

        # Status badge
        self.lbl_status = QLabel("  Disconnected  ")
        self._apply_status_style(False)
        layout.addWidget(self.lbl_status)

    # ── Helpers ────────────────────────────────────────────────────────────

    def get_port(self) -> str:
        return self.combo_port.currentData() or self.combo_port.currentText().split(" - ")[0].strip()

    def get_baudrate(self) -> int:
        return self.spin_baud.value()

    def populate_ports(self):
        self.combo_port.clear()
        for device, desc in SerialReaderThread.available_ports():
            self.combo_port.addItem(f"{device} - {desc}", device)
        if self.combo_port.count() == 0:
            self.combo_port.addItem("/dev/ttyUSB0")

    # ── State setters (called by MainWindow) ──────────────────────────────

    def set_connected(self, connected: bool):
        self._connected = connected
        self._apply_connect_style(connected)
        self._apply_status_style(connected)

    def set_simulating(self, active: bool):
        self._simulating = active
        self._apply_sim_style(active)
        self._apply_status_style(active)

    def set_model_name(self, name: str):
        self.lbl_model.setText(name)
        self.lbl_model.setStyleSheet(f"color: {Colors.SUCCESS}; font-weight: bold;")

    # ── Click handlers ────────────────────────────────────────────────────

    def _on_connect_clicked(self):
        if self._connected:
            self.disconnect_requested.emit()
        else:
            self.connect_requested.emit(self.get_port(), self.get_baudrate())

    def _on_sim_clicked(self):
        if self._simulating:
            self.stop_sim_requested.emit()
        else:
            self.simulate_requested.emit()

    # ── Styling ───────────────────────────────────────────────────────────

    def _apply_connect_style(self, connected: bool):
        if connected:
            self.btn_connect.setText("Disconnect")
            self.btn_connect.setStyleSheet(
                f"background-color: {Colors.DANGER}; color: white; font-weight: bold; padding: 4px 16px;"
            )
        else:
            self.btn_connect.setText("Connect")
            self.btn_connect.setStyleSheet(
                f"background-color: {Colors.PRIMARY}; color: white; font-weight: bold; padding: 4px 16px;"
            )

    def _apply_sim_style(self, active: bool):
        if active:
            self.btn_sim.setText("Stop Sim")
            self.btn_sim.setStyleSheet(
                f"background-color: {Colors.DANGER}; color: white; font-weight: bold; padding: 4px 12px;"
            )
        else:
            self.btn_sim.setText("Simulate")
            self.btn_sim.setStyleSheet(
                f"background-color: {Colors.SIMULATE}; color: white; font-weight: bold; padding: 4px 12px;"
            )

    def _apply_status_style(self, connected: bool):
        if connected:
            self.lbl_status.setText("  Connected  ")
            self.lbl_status.setStyleSheet(
                f"background-color: {Colors.STATUS_CONNECTED_BG}; color: white; "
                "font-weight: bold; padding: 4px 8px; border-radius: 3px;"
            )
        else:
            self.lbl_status.setText("  Disconnected  ")
            self.lbl_status.setStyleSheet(
                f"background-color: {Colors.STATUS_DISCONNECTED_BG}; color: white; "
                "font-weight: bold; padding: 4px 8px; border-radius: 3px;"
            )
