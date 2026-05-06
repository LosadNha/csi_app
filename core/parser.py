"""Parse a raw serial line into a CSIPacket."""

import csv
import json
import numpy as np
from io import StringIO
from datetime import datetime
from typing import Optional

from .models import CSIPacket, DATA_COLUMNS_STANDARD, DATA_COLUMNS_C5C6


def parse_csi_line(line: str) -> Optional[CSIPacket]:
    """Convert one CSV line (as printed by ESP-CSI firmware) into a CSIPacket.

    Returns None for non-CSI lines or malformed data.
    """
    if "CSI_DATA" not in line:
        return None

    try:
        fields = next(csv.reader(StringIO(line)))
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

    # ESP-CSI stores pairs as [imag, real, imag, real, …]
    num_sub = csi_len // 2
    csi_complex = np.empty(num_sub, dtype=np.complex64)
    for i in range(num_sub):
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
    else:  # c5c6
        pkt.device_format = "c5c6"
        pkt.seq_id = int(fields[1])
        pkt.mac = fields[2]
        pkt.rssi = int(fields[3])
        pkt.noise_floor = int(fields[5])
        pkt.channel = int(fields[8])
        pkt.local_timestamp = int(fields[9])
        pkt.sig_len = int(fields[10])

    return pkt
