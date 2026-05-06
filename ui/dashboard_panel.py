"""Right-side dashboard: stats, per-device breakdown, stacked bar chart, log."""

import json
import numpy as np
from datetime import datetime
from typing import Dict, List

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QGroupBox, QLabel,
    QTextEdit, QHBoxLayout, QLineEdit,
)
from PyQt5.QtCore import pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .style import Colors, Fonts


# ── Stacked bar chart ─────────────────────────────────────────────────────────

class StatsBarChart(FigureCanvas):
    """Packets per device per minute, stacked bars."""

    def __init__(self, parent=None, width=5, height=2.5, dpi=80):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.patch.set_facecolor(Colors.BG_DASHBOARD)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.fig = fig
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
        w = 0.6
        bottom = np.zeros(len(self.time_buckets))

        for i, (dev, counts) in enumerate(self.device_counts.items()):
            c = Colors.CHART_PALETTE[i % len(Colors.CHART_PALETTE)]
            self.ax.bar(x, counts, w, bottom=bottom, label=dev, color=c, alpha=0.85)
            bottom += np.array(counts)

        self.ax.set_xticks(x)
        self.ax.set_xticklabels(self.time_buckets, rotation=45, ha="right", fontsize=7)
        self.ax.set_title("Packets per Device per Minute", fontsize=9)
        self.ax.legend(loc="upper left", fontsize=6, ncol=2)
        self.ax.grid(axis="y", linestyle="--", alpha=0.4)
        self.ax.set_facecolor(Colors.BG_DASHBOARD)
        self.fig.tight_layout()
        self.draw()


# ── Dashboard panel ───────────────────────────────────────────────────────────

class DashboardPanel(QWidget):
    """System stats + chart + serial log + label-map editor."""

    label_map_changed = pyqtSignal(str)  # raw JSON text

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Stats group ───────────────────────────────────────────────────
        stats_group = QGroupBox("System Dashboard")
        stats_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        stats_layout = QVBoxLayout(stats_group)

        grid = QGridLayout()
        mono = Fonts.mono(10, bold=True)

        self.lbl_date = QLabel("Date: --")
        self.lbl_runtime = QLabel("Running Time: 00:00:00")
        self.lbl_total = QLabel("Total Packets: 0")
        self.lbl_devices = QLabel("Unique Devices: 0")
        self.lbl_model_info = QLabel("Model: None")
        self.lbl_subcarriers = QLabel("Subcarriers: --")

        for i, lbl in enumerate([
            self.lbl_date, self.lbl_runtime, self.lbl_total,
            self.lbl_devices, self.lbl_model_info, self.lbl_subcarriers,
        ]):
            lbl.setFont(mono)
            grid.addWidget(lbl, i // 2, i % 2)

        stats_layout.addLayout(grid)

        self.lbl_breakdown = QLabel("")
        self.lbl_breakdown.setFont(Fonts.mono(9))
        self.lbl_breakdown.setWordWrap(True)
        stats_layout.addWidget(self.lbl_breakdown)

        self.chart = StatsBarChart()
        stats_layout.addWidget(self.chart)
        root.addWidget(stats_group, stretch=3)

        # ── Log group ─────────────────────────────────────────────────────
        log_group = QGroupBox("Serial Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(Fonts.mono(8))
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)

        lm_row = QHBoxLayout()
        lm_row.addWidget(QLabel("Label Map (JSON):"))
        self.edit_label_map = QLineEdit()
        self.edit_label_map.setPlaceholderText('e.g. {"0": "Device_A", "1": "Device_B"}')
        self.edit_label_map.setFont(Fonts.mono(8))
        self.edit_label_map.editingFinished.connect(
            lambda: self.label_map_changed.emit(self.edit_label_map.text())
        )
        lm_row.addWidget(self.edit_label_map)
        log_layout.addLayout(lm_row)

        root.addWidget(log_group, stretch=1)

    # ── Public update helpers ─────────────────────────────────────────────

    def set_date(self, text: str):
        self.lbl_date.setText(f"Date: {text}")

    def set_runtime(self, h: int, m: int, s: int):
        self.lbl_runtime.setText(f"Running Time: {h:02}:{m:02}:{s:02}")

    def set_total_packets(self, n: int):
        self.lbl_total.setText(f"Total Packets: {n}")

    def set_unique_devices(self, n: int):
        self.lbl_devices.setText(f"Unique Devices: {n}")

    def set_model_info(self, text: str):
        self.lbl_model_info.setText(f"Model: {text}")

    def set_subcarriers(self, n: int):
        self.lbl_subcarriers.setText(f"Subcarriers: {n}")

    def set_device_breakdown(self, counts: Dict[str, int]):
        self.lbl_breakdown.setText(
            "  ".join(f"[{k}: {v}]" for k, v in counts.items())
        )

    def append_log(self, msg: str):
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def add_chart_point(self, time_label: str, device_name: str):
        self.chart.add_packet(time_label, device_name)
