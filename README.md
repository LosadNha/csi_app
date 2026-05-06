# RF Fingerprint — CSI Monitor

A real-time Wi-Fi **Channel State Information (CSI)** monitoring application
built with PyQt5. It connects to an ESP32 receiver board over serial, ingests
CSI frames from remote ESP transmitter devices, visualises amplitude/phase
waveforms, displays per-packet metadata, and optionally classifies the
transmitting device using a user-supplied ONNX model.

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Usage Guide](#usage-guide)
   - [Connecting to an ESP32](#connecting-to-an-esp32)
   - [Simulation Mode](#simulation-mode)
   - [Loading an ONNX Classifier](#loading-an-onnx-classifier)
   - [Label Map](#label-map)
8. [CSI Data Format](#csi-data-format)
9. [ONNX Model Requirements](#onnx-model-requirements)
10. [Bundled RFNet Model](#bundled-rfnet-model)
11. [Exporting Your Own Model](#exporting-your-own-model)
12. [Legacy Files](#legacy-files)

---

## Features

| Feature | Description |
|---------|-------------|
| **Serial CSI Ingestion** | Reads CSI_DATA lines from an ESP32 at 921600 baud (configurable). Supports both standard ESP32 and ESP32-C5/C6/C61 CSV formats. |
| **Real-time Visualisation** | Each packet is rendered as a card with side-by-side amplitude and phase plots plus a full metadata panel. |
| **ONNX Classification** | Load any `.onnx` model at runtime. The app auto-adapts to the model's input shape and shows the predicted device label and confidence on every packet. |
| **Dashboard** | Running time, total packets, unique device count, per-device breakdown, and a stacked bar chart of packets per device per minute. |
| **CSV Capture** | Every serial session is automatically saved to a timestamped CSV file. |
| **Simulation Mode** | Built-in synthetic packet generator for testing the UI and model pipeline without hardware. |

---

## Architecture

The application follows a strict **frontend / backend separation**:

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
│                   (entry point, palette)                     │
└───────────────────────────┬─────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
   ┌────────▼────────┐           ┌──────────▼──────────┐
   │     core/       │           │       ui/           │
   │   (backend)     │◄──────────│    (frontend)       │
   │                 │  imports  │                     │
   │  models.py      │           │  style.py           │
   │  parser.py      │           │  connection_panel.py│
   │  classifier.py  │           │  packet_view.py     │
   │  serial_reader.py           │  dashboard_panel.py │
   │  simulator.py   │           │  main_window.py     │
   └─────────────────┘           └─────────────────────┘
```

### Design principles

- **`core/` contains zero UI widget imports.** Data models, parsing, ONNX
  inference, and serial I/O live here. `QThread` and `pyqtSignal` are used
  only as a transport adapter for thread-safe event delivery — no widgets.
- **`ui/` imports from `core/`, never the reverse.** This one-way dependency
  keeps the backend testable and reusable independently of PyQt.
- **`style.py` is the single source of truth** for every colour token, font
  factory, and the application palette. No magic colour strings in widgets.
- **`ConnectionPanel` emits intents, not actions.** It fires signals like
  `connect_requested(port, baud)`. The orchestrator (`MainWindow`) decides
  what to do.
- **`DashboardPanel` exposes setter methods** (`set_total_packets(n)`,
  `append_log(msg)`, …). `MainWindow` calls these; the dashboard never
  reaches into backend state.
- **`MainWindow` is a thin orchestrator** (~250 lines). It wires signals
  from core threads to UI panels, handles file dialogs, and manages the
  application lifecycle. It does not render widgets or contain business logic.

---

## Project Structure

```
RF-Mockup/
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── core/                       # Backend (no widget imports)
│   ├── __init__.py             # Re-exports public API
│   ├── models.py               # CSIPacket dataclass, protocol constants
│   ├── parser.py               # parse_csi_line() — pure function
│   ├── classifier.py           # OnnxClassifier — ONNX inference wrapper
│   ├── serial_reader.py        # SerialReaderThread + port enumeration
│   └── simulator.py            # SimulationThread (synthetic packets)
│
├── ui/                         # Frontend (PyQt5 widgets)
│   ├── __init__.py             # Re-exports MainWindow
│   ├── style.py                # Colors, Fonts, make_palette()
│   ├── connection_panel.py     # Top toolbar (port, baud, connect, model)
│   ├── packet_view.py          # CSIPlotCanvas, CSIPacketWidget, PacketStreamPanel
│   ├── dashboard_panel.py      # StatsBarChart, stats grid, log, label-map
│   └── main_window.py          # MainWindow orchestrator
│
├── csi/                        # Sample CSI datasets (CSV)
│   ├── device_1/               # 22 CSV files
│   └── device_2/               # 22 CSV files
│
├── rfnet_classifier.onnx       # Pretrained RFNet ONNX classifier (bundled)
├── label_map.json              # Device label map for the ONNX model
├── best_rfnet_model.pth        # PyTorch backbone weights
├── export_onnx.py              # Script to re-export ONNX from weights + data
├── train.py                    # RFNet training script (VarCon loss)
│
├── UI.py                       # (legacy) Original mockup — kept for reference
├── app.py                      # (legacy) First refactored monolith — kept for reference
└── dummy.py                    # (legacy) Dataset loader utility
```

---

## Prerequisites

- **Python** ≥ 3.9
- **OS**: Linux, macOS, or Windows
- **Hardware** (optional): An ESP32 board flashed with
  [esp-csi](https://github.com/espressif/esp-csi) `csi_recv` firmware,
  connected via USB-to-serial adapter.

---

## Installation

```bash
# 1. Clone or unzip the project
cd RF-Mockup

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependency overview

| Package | Purpose |
|---------|---------|
| `PyQt5` | GUI framework |
| `numpy` | Numerical operations on CSI arrays |
| `matplotlib` | Amplitude / phase plots |
| `pyserial` | Serial port communication with ESP32 |
| `onnxruntime` | ONNX model inference for device classification |
| `scipy` | Signal processing utilities |
| `pandas` | CSV data handling (used by legacy `dummy.py`) |

> `pyserial` and `onnxruntime` are optional. The app gracefully degrades
> when they are missing — serial connection and model loading will show
> an informative error message.

---

## Quick Start

```bash
python main.py
```

The GUI window opens. From here you can either:

- **Connect** to a real ESP32 via the serial port dropdown, or
- Click **Simulate** to generate synthetic packets without hardware.

---

## Usage Guide

### Connecting to an ESP32

1. Plug the ESP32 receiver board into a USB port.
2. Click **Refresh** to scan available serial ports.
3. Select the correct port from the dropdown (e.g. `/dev/ttyUSB0`).
4. Set the baud rate (default `921600` matches the esp-csi firmware).
5. Click **Connect**.

The status badge turns green and CSI packets start streaming into the
**Live CSI Stream** panel on the left. Each card shows:

- Dual amplitude / phase waveform plots
- Full metadata: MAC address, RSSI, channel, noise floor, signal mode,
  MCS index, bandwidth, subcarrier count, sequence ID, signal length,
  and receive timestamp.

A timestamped CSV file (`csi_capture_YYYYMMDD_HHMMSS.csv`) is
automatically saved in the project directory.

Click **Disconnect** to stop.

### Simulation Mode

Click **Simulate** to start generating synthetic CSI packets from three
fake MAC addresses. This is useful for:

- Testing the UI layout and scrolling performance.
- Verifying that an ONNX model loads and produces predictions.
- Demonstrating the application without hardware.

Click **Stop Sim** to stop.

### Loading an ONNX Classifier

1. Click **Load Model…** and select a `.onnx` file.
2. The model metadata (input name, shape) is printed to the **Serial Log**.
3. From this point on, every incoming packet is classified. The predicted
   device label and confidence percentage appear on each packet card.
4. Cards are colour-coded by confidence:
   - **Green border** — confidence ≥ 70%
   - **Yellow border** — confidence 40–70%
   - **Red border** — confidence < 40%

### Label Map

By default, predictions are shown as `Device_0`, `Device_1`, etc.
To assign human-readable names, type a JSON object into the
**Label Map** field:

```json
{"0": "ESP32_Kitchen", "1": "ESP32_Garage", "2": "ESP32_Office"}
```

Press Enter to apply. The label map is used immediately for all
subsequent packets.

---

## CSI Data Format

The application supports two CSV line formats, both printed by the
[esp-csi](https://github.com/espressif/esp-csi) firmware over serial:

### Standard ESP32 (ESP32 / S2 / S3 / C3) — 25 fields

```
CSI_DATA,<id>,<mac>,<rssi>,<rate>,<sig_mode>,<mcs>,<bandwidth>,
<smoothing>,<not_sounding>,<aggregation>,<stbc>,<fec_coding>,
<sgi>,<noise_floor>,<ampdu_cnt>,<channel>,<secondary_channel>,
<local_timestamp>,<ant>,<sig_len>,<rx_state>,<len>,<first_word>,<data>
```

### ESP32-C5 / C6 / C61 — 15 fields

```
CSI_DATA,<seq>,<mac>,<rssi>,<rate>,<noise_floor>,<fft_gain>,
<agc_gain>,<channel>,<local_timestamp>,<sig_len>,<rx_state>,
<len>,<first_word>,<data>
```

The `<data>` field is a JSON array of integers arranged as
`[imag₀, real₀, imag₁, real₁, …]`. Each pair represents one OFDM
subcarrier as a complex number. Common subcarrier counts:

| Bandwidth | Subcarriers | Array length |
|-----------|-------------|-------------|
| HT20      | 52          | 104         |
| HT40      | 106         | 212         |
| VHT80     | 114         | 228         |

---

## ONNX Model Requirements

The classifier expects an ONNX model whose **single input tensor**
accepts CSI features. The application automatically adapts to these
common input layouts:

| Input shape | Feature layout |
|-------------|---------------|
| `(batch, N_sub, 2)` | `[amplitude, phase]` per subcarrier |
| `(batch, 2, N_sub)` | Transposed — auto-detected |
| `(batch, N_sub)` | Amplitude only |

The **output** should be a logits vector (one element per class).
A softmax is applied internally to obtain probabilities.

If the model's subcarrier dimension does not match the incoming data,
the application pads with zeros or truncates to fit.

---

## Bundled RFNet Model

The package ships with a pretrained model ready to use:

| File | Description |
|------|-------------|
| `rfnet_classifier.onnx` | End-to-end classifier (StandardScaler + RFNet backbone + centroid similarity) |
| `label_map.json` | `{"0": "device_1", "1": "device_2"}` |
| `best_rfnet_model.pth` | Raw PyTorch weights for the RFNet backbone |

### About the model

**RFNet** is a gated depthwise-separable Conv1D network trained with
variational contrastive loss (VarCon). It produces 128-dimensional
L2-normalised embeddings. Classification is performed via cosine
similarity against per-device support centroids.

The ONNX wrapper bakes in three stages so no preprocessing is needed:

```
raw (batch, 2, 117) → StandardScaler → RFNet backbone → cosine sim → logits
```

| Property | Value |
|----------|-------|
| Input | `csi_input` — shape `(batch, 2, 117)` float32 |
| Output | `logits` — shape `(batch, 2)` float32 |
| Subcarriers | 117 (auto-padded/truncated by the app) |
| Classes | 2 (`device_1`, `device_2`) |

### Using the bundled model

1. Launch `python main.py`.
2. Click **Load Model…** → select `rfnet_classifier.onnx`.
3. Paste the label map into the text field:
   ```json
   {"0": "device_1", "1": "device_2"}
   ```
4. Start **Simulate** or **Connect** — each packet card now shows
   the predicted device and confidence.

---

## Exporting Your Own Model

Use `export_onnx.py` to re-export with different data or weights:

```bash
python export_onnx.py
```

The script:

1. Loads the RFNet backbone from `best_rfnet_model.pth`.
2. Reads all CSI CSV files from `csi/<device_name>/`.
3. Fits a `StandardScaler` on the combined data.
4. Computes per-device support centroids in embedding space.
5. Wraps backbone + scaler + centroids into a single ONNX graph.
6. Writes `rfnet_classifier.onnx` and `label_map.json`.

To add more devices, create folders under `csi/` with CSV files in
the same format (columns: `Sub_X_Real`, `Sub_X_Imag` for 0 ≤ X ≤ 116)
and re-run the export.

To use a different backbone checkpoint, replace `best_rfnet_model.pth`
or edit the `WEIGHTS` variable in the script.

---

## Legacy Files

| File | Description |
|------|-------------|
| `UI.py` | Original mockup with synthetic data and anti-spoofing simulation. |
| `app.py` | First refactored monolith before the frontend/backend split. |
| `dummy.py` | Dataset loader that reads the `csi/` CSV files into numpy arrays. |

These files are kept for reference and are not required to run the
application.
