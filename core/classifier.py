"""ONNX-based CSI classifier for device identification."""

import numpy as np
from typing import Dict, Optional, Tuple

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class OnnxClassifier:
    """Load an ONNX model and classify CSI feature vectors."""

    def __init__(self):
        self.session: Optional["ort.InferenceSession"] = None
        self.model_path: str = ""
        self.input_name: str = ""
        self.input_shape: tuple = ()
        self.label_map: Dict[int, str] = {}

    # -- public API --

    def load(self, model_path: str, label_map: Dict[int, str] | None = None):
        if not HAS_ONNX:
            raise RuntimeError(
                "onnxruntime is not installed. Install with: pip install onnxruntime"
            )
        self.session = ort.InferenceSession(model_path)
        self.model_path = model_path
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = tuple(inp.shape)
        self.label_map = label_map or {}

    def predict(self, amplitude: np.ndarray, phase: np.ndarray) -> Tuple[str, float]:
        """Return (predicted_label, confidence_percent) for one CSI sample."""
        if self.session is None:
            return ("N/A", 0.0)

        features = self._prepare_features(amplitude, phase)

        try:
            logits = self.session.run(None, {self.input_name: features})[0][0]
            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / exp_l.sum()
            idx = int(np.argmax(probs))
            return (self.label_map.get(idx, f"Device_{idx}"), float(probs[idx]) * 100.0)
        except Exception as e:
            return (f"Error: {e}", 0.0)

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

    # -- internals --

    def _prepare_features(self, amplitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """Shape the CSI amplitude+phase into the tensor the model expects.

        Handles three common ONNX input layouts:
          (batch, 2, N_sub)   — channels-first  (e.g. RFNet)
          (batch, N_sub, 2)   — channels-last
          (batch, N_sub)      — amplitude only
        Dynamic axes (strings like 'batch_size') are treated as wildcards.
        """
        n = len(amplitude)
        expected = self.input_shape
        ndim = len(expected)

        def _is(val, target):
            """True if val equals target or is a dynamic-axis string."""
            return val == target or isinstance(val, str)

        if ndim == 3:
            d1, d2 = expected[1], expected[2]

            # Decide layout
            if (d1 == 2 or _is(d1, 2)) and _is(d2, n):
                # Model wants (batch, 2, N_sub)
                features = np.stack([amplitude, phase], axis=0)         # (2, N)
            elif _is(d1, n) and (d2 == 2 or _is(d2, 2)):
                # Model wants (batch, N_sub, 2)
                features = np.stack([amplitude, phase], axis=-1)        # (N, 2)
            else:
                # Unknown — default to channels-first
                features = np.stack([amplitude, phase], axis=0)

            features = features.astype(np.float32)[np.newaxis, ...]     # (1, ?, ?)

            # Pad / truncate subcarrier axis to match a fixed dimension
            sub_axis = 2 if features.shape[1] == 2 else 1
            target_sub = expected[sub_axis]
            if isinstance(target_sub, int) and target_sub > 2:
                cur = features.shape[sub_axis]
                if cur < target_sub:
                    pad = [(0, 0)] * features.ndim
                    pad[sub_axis] = (0, target_sub - cur)
                    features = np.pad(features, pad, mode="constant")
                elif cur > target_sub:
                    slices = [slice(None)] * features.ndim
                    slices[sub_axis] = slice(0, target_sub)
                    features = features[tuple(slices)]

        elif ndim == 2:
            features = amplitude[np.newaxis, :].astype(np.float32)
        else:
            features = np.stack([amplitude, phase], axis=0).astype(np.float32)
            features = features[np.newaxis, ...]

        return features
