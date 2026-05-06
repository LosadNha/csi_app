"""Core backend: data models, CSI parsing, serial I/O, classification.

No PyQt widget imports allowed in this package.
QThread/pyqtSignal are used only as a transport adapter in serial_reader/simulator.
"""

from .models import CSIPacket, DATA_COLUMNS_STANDARD, DATA_COLUMNS_C5C6
from .parser import parse_csi_line
from .classifier import OnnxClassifier
from .serial_reader import SerialReaderThread
from .simulator import SimulationThread
