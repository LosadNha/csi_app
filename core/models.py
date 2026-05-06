"""Data models and protocol constants for ESP-CSI packets."""

from dataclasses import dataclass, field
import numpy as np

# CSV column names for standard ESP32 (ESP32/S2/S3/C3)
DATA_COLUMNS_STANDARD = [
    'type', 'id', 'mac', 'rssi', 'rate', 'sig_mode', 'mcs', 'bandwidth',
    'smoothing', 'not_sounding', 'aggregation', 'stbc', 'fec_coding',
    'sgi', 'noise_floor', 'ampdu_cnt', 'channel', 'secondary_channel',
    'local_timestamp', 'ant', 'sig_len', 'rx_state', 'len', 'first_word', 'data',
]

# CSV column names for ESP32-C5/C6/C61
DATA_COLUMNS_C5C6 = [
    'type', 'seq', 'mac', 'rssi', 'rate', 'noise_floor', 'fft_gain',
    'agc_gain', 'channel', 'local_timestamp', 'sig_len', 'rx_state',
    'len', 'first_word', 'data',
]

# Human-readable lookup tables
SIG_MODE_NAMES = {0: "Legacy", 1: "HT", 2: "VHT"}
BANDWIDTH_NAMES = {0: "HT20", 1: "HT40"}


@dataclass
class CSIPacket:
    """A single parsed CSI frame with metadata and complex-valued subcarriers."""

    # PC-side receive time
    timestamp: str

    # RF metadata
    mac: str = ""
    rssi: int = 0
    channel: int = 0
    noise_floor: int = 0
    sig_mode: int = -1
    mcs: int = -1
    bandwidth: int = -1
    sig_len: int = 0
    seq_id: int = 0
    local_timestamp: int = 0

    # CSI payload
    csi_len: int = 0
    csi_complex: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.complex64))
    amplitude: np.ndarray = field(default_factory=lambda: np.array([]))
    phase: np.ndarray = field(default_factory=lambda: np.array([]))

    # Bookkeeping
    raw_line: str = ""
    device_format: str = "standard"       # "standard" | "c5c6"

    # Classification results (filled by the classifier, not the parser)
    predicted_device: str = "N/A"
    prediction_confidence: float = 0.0

    # -- convenience helpers --

    @property
    def num_subcarriers(self) -> int:
        return len(self.amplitude)

    @property
    def sig_mode_name(self) -> str:
        return SIG_MODE_NAMES.get(self.sig_mode, str(self.sig_mode))

    @property
    def bandwidth_name(self) -> str:
        return BANDWIDTH_NAMES.get(self.bandwidth, str(self.bandwidth))
