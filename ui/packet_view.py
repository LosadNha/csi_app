"""Widgets for displaying individual CSI packets and the live stream list."""

import numpy as np

from PyQt5.QtWidgets import (
    QFrame, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QScrollArea,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.models import CSIPacket
from .style import Colors, Fonts


# ── Single-packet plot ────────────────────────────────────────────────────────

class CSIPlotCanvas(FigureCanvas):
    """Side-by-side amplitude + phase mini-plots."""

    def __init__(self, parent=None, width=6, height=2, dpi=80, bg_color=Colors.BG_CARD):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.patch.set_facecolor(bg_color)
        self.axes_amp = fig.add_subplot(121)
        self.axes_phase = fig.add_subplot(122)
        super().__init__(fig)
        self.setParent(parent)
        self.fig = fig

    def plot(self, amplitude: np.ndarray, phase: np.ndarray):
        x = np.arange(len(amplitude))

        for ax, data, color, title, ylabel in [
            (self.axes_amp,   amplitude, Colors.PLOT_AMP,   "Amplitude", "Magnitude"),
            (self.axes_phase, phase,     Colors.PLOT_PHASE, "Phase",     "Angle (rad)"),
        ]:
            ax.clear()
            ax.plot(x, data, color=color, linewidth=1.0, alpha=0.9)
            ax.set_title(title, fontsize=8, fontweight="bold")
            ax.set_xlabel("Subcarrier", fontsize=6)
            ax.set_ylabel(ylabel, fontsize=6)
            ax.tick_params(axis="both", labelsize=5)
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.patch.set_alpha(0.0)

        self.fig.tight_layout()
        self.draw()


# ── Single-packet card ────────────────────────────────────────────────────────

class CSIPacketWidget(QFrame):
    """Card showing one CSI packet: plots on the left, metadata on the right."""

    def __init__(self, pkt: CSIPacket, show_prediction: bool = True):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setLineWidth(1)
        self.setFixedHeight(190)

        bg, border = self._pick_colors(pkt, show_prediction)
        self.setStyleSheet(
            f"CSIPacketWidget {{ background-color: {bg}; "
            f"border: 1px solid {border}; border-radius: 4px; }}"
        )

        root = QHBoxLayout()
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # -- plots --
        canvas = CSIPlotCanvas(bg_color=bg)
        canvas.plot(pkt.amplitude, pkt.phase)
        canvas.setFixedWidth(420)
        root.addWidget(canvas, stretch=0)

        # -- metadata --
        info = QWidget()
        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(2, 2, 2, 2)
        info_layout.setSpacing(2)

        # prediction header
        pred_lbl = QLabel()
        pred_lbl.setFont(Fonts.mono(9, bold=True))
        if show_prediction and pkt.predicted_device != "N/A":
            pred_lbl.setText(
                f"Predicted: {pkt.predicted_device}  ({pkt.prediction_confidence:.1f}%)"
            )
        else:
            pred_lbl.setText("Predicted: N/A (no model)")
            pred_lbl.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        info_layout.addWidget(pred_lbl)

        # metadata grid
        meta = [
            ("Time",        pkt.timestamp),
            ("MAC",         pkt.mac),
            ("RSSI",        f"{pkt.rssi} dBm"),
            ("Channel",     str(pkt.channel)),
            ("Noise Floor", f"{pkt.noise_floor} dBm"),
            ("Subcarriers", str(pkt.num_subcarriers)),
            ("Seq ID",      str(pkt.seq_id)),
            ("Sig Len",     str(pkt.sig_len)),
        ]
        if pkt.device_format == "standard":
            meta += [
                ("Sig Mode",  pkt.sig_mode_name),
                ("MCS",       str(pkt.mcs)),
                ("Bandwidth", pkt.bandwidth_name),
            ]

        grid = QGridLayout()
        grid.setSpacing(1)
        mono = Fonts.mono(8)
        for i, (key, val) in enumerate(meta):
            r, c = divmod(i, 2)
            kl = QLabel(f"{key}:")
            kl.setFont(mono)
            kl.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
            vl = QLabel(val)
            vl.setFont(mono)
            vl.setStyleSheet("font-weight: bold;")
            grid.addWidget(kl, r, c * 2)
            grid.addWidget(vl, r, c * 2 + 1)

        info_layout.addLayout(grid)
        info_layout.addStretch()

        root.addWidget(info, stretch=1)
        self.setLayout(root)

    @staticmethod
    def _pick_colors(pkt: CSIPacket, show_pred: bool):
        if show_pred and pkt.predicted_device != "N/A":
            if pkt.prediction_confidence >= 70:
                return Colors.CARD_OK, Colors.BORDER_OK
            elif pkt.prediction_confidence >= 40:
                return Colors.CARD_WARN, Colors.BORDER_WARN
            else:
                return Colors.CARD_ERR, Colors.BORDER_ERR
        return Colors.BG_CARD, Colors.BORDER_NEUTRAL


# ── Scrollable packet stream ─────────────────────────────────────────────────

MAX_VISIBLE_PACKETS = 150


class PacketStreamPanel(QWidget):
    """A scrollable, newest-on-top list of CSIPacketWidgets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel("Live CSI Stream")
        lbl.setFont(Fonts.ui(12, bold=True))
        layout.addWidget(lbl)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._content = QWidget()
        self._layout = QVBoxLayout(self._content)
        self._layout.setAlignment(Qt.AlignTop)
        self._layout.setSpacing(4)
        self._scroll.setWidget(self._content)
        layout.addWidget(self._scroll)

    def add_packet(self, pkt: CSIPacket, show_prediction: bool):
        widget = CSIPacketWidget(pkt, show_prediction)
        self._layout.insertWidget(0, widget)

        # Evict oldest to cap memory
        while self._layout.count() > MAX_VISIBLE_PACKETS:
            item = self._layout.takeAt(self._layout.count() - 1)
            if item and item.widget():
                item.widget().deleteLater()
