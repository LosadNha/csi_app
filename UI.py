import sys
import os
import random
import time
import numpy as np
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QScrollArea, 
                             QFrame, QSplitter, QPushButton, QProgressBar)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ==========================================
# 1. DATA PROCESSING & MOCK LOGIC
# ==========================================

class DataProcessor:
    @staticmethod
    def read_csi_waveform(file_path=None, mock_type=None):
        """
        Simulates reading CSI data. 
        Returns: subcarriers (x-axis), amplitude, phase
        """
        subcarriers = np.arange(-59, 58) # 117 subcarriers from -59 to +58 excluding 0

        # GENERATE SYNTHETIC DATA
        noise_level = 0.5 if mock_type == 'genuine' else 2.0
        base_signal = np.sin(subcarriers * 0.2) 
        
        real_part = base_signal + np.random.normal(0, noise_level, 117)
        imag_part = np.cos(subcarriers * 0.2) + np.random.normal(0, noise_level, 117)
        csi_complex = real_part + 1j * imag_part
        
        amplitude = np.abs(csi_complex)
        phase = np.unwrap(np.angle(csi_complex))
        
        return subcarriers, amplitude, phase

# ==========================================
# 2. CUSTOM UI WIDGETS
# ==========================================

class MiniPlotCanvas(FigureCanvas):
    """A small Matplotlib canvas for the side-by-side Amp/Phase plots"""
    def __init__(self, parent=None, width=5, height=2.5, dpi=80, bg_color='#f0f0f0'):
        # Increased height slightly to accommodate axis labels
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.patch.set_facecolor(bg_color)
        self.axes_amp = fig.add_subplot(121)
        self.axes_phase = fig.add_subplot(122)
        super(MiniPlotCanvas, self).__init__(fig)
        self.setParent(parent)
        self.fig = fig
        # tight_layout called initially to set margins
        self.fig.tight_layout()

    def update_plots(self, x, amp, phase):
        # --- Amplitude Plot ---
        self.axes_amp.clear()
        self.axes_amp.plot(x, amp, color='#2c3e50', linewidth=1.5)
        self.axes_amp.set_title("Amplitude", fontsize=9, fontweight='bold')
        
        # Axis Labels & Ticks
        self.axes_amp.set_xlabel("Subcarrier", fontsize=7)
        self.axes_amp.set_ylabel("Magnitude", fontsize=7)
        self.axes_amp.tick_params(axis='both', which='major', labelsize=6)
        
        # Styling
        self.axes_amp.grid(True, linestyle=':', alpha=0.6)
        self.axes_amp.patch.set_alpha(0.0) 

        # --- Phase Plot ---
        self.axes_phase.clear()
        self.axes_phase.plot(x, phase, color='#2c3e50', linewidth=1.5)
        self.axes_phase.set_title("Phase", fontsize=9, fontweight='bold')
        
        # Axis Labels & Ticks
        self.axes_phase.set_xlabel("Subcarrier", fontsize=7)
        self.axes_phase.set_ylabel("Angle (rad)", fontsize=7)
        self.axes_phase.tick_params(axis='both', which='major', labelsize=6)
        
        # Styling
        self.axes_phase.grid(True, linestyle=':', alpha=0.6)
        self.axes_phase.patch.set_alpha(0.0)
        
        # Adjust layout to prevent clipping of labels
        self.fig.tight_layout()
        self.draw()

class PacketItemWidget(QFrame):
    """Represents a single row in Section A or B."""
    def __init__(self, x, amp, phase, score=None, time_str=None, is_monitoring=False):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setLineWidth(1)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        bg_color = "#E0E0E0" # Default Grey (Section A)
        
        if is_monitoring:
            if score >= 50:
                bg_color = "#C8E6C9" # Light Green
                self.setStyleSheet(f"background-color: {bg_color}; border: 1px solid green; border-radius: 5px;")
            else:
                bg_color = "#FFCDD2" # Light Red
                self.setStyleSheet(f"background-color: {bg_color}; border: 1px solid red; border-radius: 5px;")
        else:
            self.setStyleSheet(f"background-color: {bg_color}; border: 1px solid gray; border-radius: 5px;")

        # Pass bg_color to canvas so it matches the frame
        self.canvas = MiniPlotCanvas(bg_color=bg_color)
        self.canvas.update_plots(x, amp, phase)
        layout.addWidget(self.canvas)
        
        if is_monitoring:
            info_layout = QHBoxLayout()
            score_lbl = QLabel(f"Similarity score: {score:.1f}%")
            score_lbl.setFont(QFont("Arial", 10, QFont.Bold))
            time_lbl = QLabel(f"Received Time: {time_str}")
            time_lbl.setFont(QFont("Arial", 9))
            
            info_layout.addWidget(score_lbl)
            info_layout.addStretch()
            info_layout.addWidget(time_lbl)
            layout.addLayout(info_layout)
            
        self.setLayout(layout)
        # Increased height to fit the new axis labels without cramping
        self.setFixedHeight(220) 

class BarChartCanvas(FigureCanvas):
    """Interactive Stacked Bar Chart for Section C"""
    def __init__(self, parent=None, width=5, height=3, dpi=80):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.patch.set_facecolor('#dcdcdc') 
        self.ax = fig.add_subplot(111)
        super(BarChartCanvas, self).__init__(fig)
        self.setParent(parent)
        self.fig = fig
        
        self.time_buckets = []
        self.genuine_counts = []
        self.rogue_counts = []

    def update_chart(self, time_label, is_genuine):
        if not self.time_buckets or time_label != self.time_buckets[-1]:
            self.time_buckets.append(time_label)
            self.genuine_counts.append(0)
            self.rogue_counts.append(0)
            
            if len(self.time_buckets) > 8:
                self.time_buckets.pop(0)
                self.genuine_counts.pop(0)
                self.rogue_counts.pop(0)

        if is_genuine:
            self.genuine_counts[-1] += 1
        else:
            self.rogue_counts[-1] += 1

        self.ax.clear()
        
        indices = np.arange(len(self.time_buckets))
        width = 0.5

        p1 = self.ax.bar(indices, self.genuine_counts, width, color='green', label='Genuine')
        p2 = self.ax.bar(indices, self.rogue_counts, width, bottom=self.genuine_counts, color='red', label='Rogue')

        self.ax.set_xticks(indices)
        self.ax.set_xticklabels(self.time_buckets, rotation=45, ha='right', fontsize=8)
        self.ax.set_title("Total Infiltration Attempts vs Requests", fontsize=9)
        self.ax.legend(loc='upper left', fontsize=7)
        self.ax.grid(axis='y', linestyle='--', alpha=0.5)
        self.ax.set_facecolor('#dcdcdc')
        
        self.fig.tight_layout()
        self.draw()

# ==========================================
# 3. MAIN WINDOW
# ==========================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RF Fingerprint Anti-Spoofing Mockup")
        self.resize(1200, 900) # Slightly taller for the larger plots
        
        # State Variables
        self.start_time = None
        self.total_requests = 0
        self.total_rogue = 0
        self.packet_buffer = [] 
        
        self.setup_ui()

    def setup_ui(self):
        """Builds the layout with a Start Button at the top"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Top-level Vertical Layout
        top_layout = QVBoxLayout(central_widget)

        # --- START BUTTON AREA ---
        self.btn_start = QPushButton("START DETECTION SYSTEM")
        self.btn_start.setFont(QFont("Arial", 12, QFont.Bold))
        self.btn_start.setFixedHeight(50)
        self.btn_start.setStyleSheet("background-color: #2196F3; color: white; border-radius: 5px;")
        self.btn_start.clicked.connect(self.start_system_sequence)
        top_layout.addWidget(self.btn_start)

        # --- MAIN CONTENT AREA (Columns) ---
        main_columns_widget = QWidget()
        main_columns_layout = QHBoxLayout(main_columns_widget)
        main_columns_layout.setContentsMargins(0,0,0,0)
        
        # LEFT COLUMN: Section B (Realtime Monitor)
        self.section_b_container = QWidget()
        layout_b = QVBoxLayout(self.section_b_container)
        
        lbl_b = QLabel("Realtime Monitor")
        lbl_b.setFont(QFont("Arial", 12, QFont.Bold))
        layout_b.addWidget(lbl_b)

        self.scroll_b = QScrollArea()
        self.scroll_b.setWidgetResizable(True)
        self.scroll_content_b = QWidget()
        self.scroll_layout_b = QVBoxLayout(self.scroll_content_b)
        self.scroll_layout_b.setAlignment(Qt.AlignTop) 
        self.scroll_b.setWidget(self.scroll_content_b)
        layout_b.addWidget(self.scroll_b)
        
        # RIGHT COLUMN: Splitter (A and C)
        right_splitter = QSplitter(Qt.Vertical)
        
        # Section A: Verification Support
        self.section_a_container = QWidget()
        layout_a = QVBoxLayout(self.section_a_container)
        lbl_a = QLabel("Verification Support (Prerecording at Startup)")
        lbl_a.setFont(QFont("Arial", 12, QFont.Bold))
        layout_a.addWidget(lbl_a)
        
        self.scroll_a = QScrollArea()
        self.scroll_a.setWidgetResizable(True)
        self.scroll_content_a = QWidget()
        self.scroll_layout_a = QVBoxLayout(self.scroll_content_a)
        self.scroll_a.setWidget(self.scroll_content_a)
        layout_a.addWidget(self.scroll_a)
        
        # Section C: System Dashboard
        self.section_c_container = QWidget()
        self.section_c_container.setStyleSheet("background-color: #cfd8dc; border-radius: 8px;")
        layout_c = QVBoxLayout(self.section_c_container)
        
        info_grid = QGridLayout()
        self.lbl_device = QLabel("Device Type: ESP32")
        self.lbl_date = QLabel(f"Date: --")
        self.lbl_runtime = QLabel("Running Time: 00:00:00")
        self.lbl_requests = QLabel("Total request: 0")
        self.lbl_infiltration = QLabel("Total Infiltration Attempts: 0")
        
        for l in [self.lbl_device, self.lbl_date, self.lbl_runtime, self.lbl_requests, self.lbl_infiltration]:
            l.setFont(QFont("Courier New", 10, QFont.Bold))
            
        info_grid.addWidget(self.lbl_device, 0, 0)
        info_grid.addWidget(self.lbl_date, 1, 0)
        info_grid.addWidget(self.lbl_runtime, 2, 0)
        info_grid.addWidget(self.lbl_requests, 3, 0)
        info_grid.addWidget(self.lbl_infiltration, 4, 0)
        
        layout_c.addLayout(info_grid)
        self.chart_widget = BarChartCanvas()
        layout_c.addWidget(self.chart_widget)

        right_splitter.addWidget(self.section_a_container)
        right_splitter.addWidget(self.section_c_container)
        right_splitter.setStretchFactor(0, 1) 
        right_splitter.setStretchFactor(1, 1) 
        
        main_columns_layout.addWidget(self.section_b_container, 50)
        main_columns_layout.addWidget(right_splitter, 50)
        
        top_layout.addWidget(main_columns_widget)

    def start_system_sequence(self):
        """Called when Button is clicked"""
        # 1. Update UI State
        self.btn_start.setText("SYSTEM RUNNING...")
        self.btn_start.setEnabled(False) # Prevent double clicking
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 5px;")
        
        self.start_time = datetime.now()
        self.lbl_date.setText(f"Date: {self.start_time.strftime('%Y-%m-%d')}")
        
        # 2. Run Logic
        self.initialize_dataset()
        self.start_monitoring()

    def initialize_dataset(self):
        """Load 20 random samples into Section A"""
        for _ in range(20):
            x, amp, phase = DataProcessor.read_csi_waveform(mock_type='genuine')
            item = PacketItemWidget(x, amp, phase, is_monitoring=False)
            self.scroll_layout_a.addWidget(item)

    def start_monitoring(self):
        """Start timers"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_heartbeat)
        self.timer.start(1500) 
        
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_dashboard_text)
        self.clock_timer.start(1000)

    def on_heartbeat(self):
        self.total_requests += 1
        
        is_rogue = random.choice([True, False, False])
        source_type = 'rogue' if is_rogue else 'genuine'
        
        x, amp, phase = DataProcessor.read_csi_waveform(mock_type=source_type)
        
        if source_type == 'genuine':
            score = random.uniform(75.0, 99.9)
        else:
            score = random.uniform(15.0, 40.0)
            self.total_rogue += 1
            
        time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        item = PacketItemWidget(x, amp, phase, score=score, time_str=time_str, is_monitoring=True)
        
        self.scroll_layout_b.insertWidget(0, item)
        self.packet_buffer.append(item)
        
        if self.scroll_layout_b.count() > 200:
            removed_item = self.scroll_layout_b.takeAt(200)
            if removed_item.widget():
                removed_item.widget().deleteLater()

        self.lbl_requests.setText(f"Total request: {self.total_requests}")
        self.lbl_infiltration.setText(f"Total Infiltration Attempts: {self.total_rogue}")
        
        current_minute = datetime.now().strftime("%H:%M")
        self.chart_widget.update_chart(current_minute, not is_rogue)

    def update_dashboard_text(self):
        if self.start_time:
            now = datetime.now()
            delta = now - self.start_time
            seconds = int(delta.total_seconds())
            h, remainder = divmod(seconds, 3600)
            m, s = divmod(remainder, 60)
            self.lbl_runtime.setText(f"Running Time: {h:02}:{m:02}:{s:02}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())