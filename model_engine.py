import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image, ImageOps


def _same_pad_2d(x: np.ndarray, kh: int, kw: int, sh: int, sw: int) -> tuple[np.ndarray, int, int]:
    h, w, _ = x.shape
    oh = (h + sh - 1) // sh
    ow = (w + sw - 1) // sw
    ph = max((oh - 1) * sh + kh - h, 0)
    pw = max((ow - 1) * sw + kw - w, 0)
    pt, pb = ph // 2, ph - ph // 2
    pl, pr = pw // 2, pw - pw // 2
    if pt or pb or pl or pr:
        x = np.pad(x, ((pt, pb), (pl, pr), (0, 0)), mode="constant")
    return x, oh, ow


def _conv2d(x: np.ndarray, w: np.ndarray, strides: tuple[int, int], padding: str) -> np.ndarray:
    kh, kw, cin, cout = w.shape
    sh, sw = strides
    if padding == "same":
        x, oh, ow = _same_pad_2d(x, kh, kw, sh, sw)
    else:
        h, w_in, _ = x.shape
        oh = (h - kh) // sh + 1
        ow = (w_in - kw) // sw + 1

    if kh == 1 and kw == 1 and sh == 1 and sw == 1:
        return (x.reshape(-1, cin) @ w.reshape(cin, cout)).reshape(oh, ow, cout)

    s0, s1, s2 = x.strides
    patches = np.lib.stride_tricks.as_strided(
        x,
        shape=(oh, ow, kh, kw, cin),
        strides=(s0 * sh, s1 * sw, s0, s1, s2),
    )
    out = patches.reshape(oh * ow, kh * kw * cin) @ w.reshape(kh * kw * cin, cout)
    return out.reshape(oh, ow, cout)


def _depthwise_conv2d(x: np.ndarray, w: np.ndarray, strides: tuple[int, int], padding: str) -> np.ndarray:
    kh, kw, cin, mult = w.shape
    if mult != 1:
        raise ValueError(f"Unsupported depth_multiplier={mult}")
    sh, sw = strides
    if padding == "same":
        x, oh, ow = _same_pad_2d(x, kh, kw, sh, sw)
    else:
        h, w_in, _ = x.shape
        oh = (h - kh) // sh + 1
        ow = (w_in - kw) // sw + 1

    s0, s1, s2 = x.strides
    patches = np.lib.stride_tricks.as_strided(
        x,
        shape=(oh, ow, kh, kw, cin),
        strides=(s0 * sh, s1 * sw, s0, s1, s2),
    )
    return np.einsum("ijxyc,xyc->ijc", patches, w[:, :, :, 0], optimize=True)


def _extract_inbound_names(layer_cfg: dict) -> list[str]:
    nodes = layer_cfg.get("inbound_nodes", [])
    if not nodes:
        return []
    args = nodes[0].get("args", [])
    names: list[str] = []
    for arg in args:
        if isinstance(arg, dict) and arg.get("class_name") == "__keras_tensor__":
            names.append(arg["config"]["keras_history"][0])
        elif isinstance(arg, list):
            for item in arg:
                if isinstance(item, dict) and item.get("class_name") == "__keras_tensor__":
                    names.append(item["config"]["keras_history"][0])
    return names


class MaskClassifier:
    """Loads a Keras MobileNetV2 binary mask classifier from an .h5 file and runs inference."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        with h5py.File(self.model_path, "r") as f:
            model_cfg = json.loads(f.attrs["model_config"])
            self.layers = model_cfg["config"]["layers"]
            mw = f["model_weights"]
            self.weights: dict[str, dict[str, np.ndarray]] = {}
            for layer in self.layers:
                name = layer["name"]
                if name in mw and name in mw[name]:
                    group = mw[name][name]
                    self.weights[name] = {
                        k: np.array(group[k], dtype=np.float32) for k in group.keys()
                    }

        input_shape = self.layers[0]["config"].get("batch_shape", [None, 224, 224, 3])
        self.img_height = int(input_shape[1] or 224)
        self.img_width = int(input_shape[2] or 224)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to normalized float32 tensor (H, W, 3) in [0, 1]."""
        img = ImageOps.exif_transpose(image).convert("RGB")
        img = img.resize((self.img_width, self.img_height), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    def predict_raw(self, arr: np.ndarray) -> float:
        """Run forward pass on preprocessed (H, W, 3) float32 image array."""
        outputs: dict[str, np.ndarray] = {}
        for layer in self.layers:
            cls = layer["class_name"]
            name = layer["name"]
            cfg = layer["config"]

            if cls == "InputLayer":
                outputs[name] = arr
                continue

            in_names = _extract_inbound_names(layer)

            if cls == "Conv2D":
                outputs[name] = _conv2d(
                    outputs[in_names[0]],
                    self.weights[name]["kernel"],
                    tuple(cfg["strides"]),
                    cfg["padding"],
                )
            elif cls == "DepthwiseConv2D":
                outputs[name] = _depthwise_conv2d(
                    outputs[in_names[0]],
                    self.weights[name]["kernel"],
                    tuple(cfg["strides"]),
                    cfg["padding"],
                )
            elif cls == "BatchNormalization":
                w = self.weights[name]
                scale = w["gamma"] / np.sqrt(w["moving_variance"] + cfg["epsilon"])
                outputs[name] = (outputs[in_names[0]] - w["moving_mean"]) * scale + w["beta"]
            elif cls == "ReLU":
                max_val = cfg.get("max_value")
                t = np.maximum(outputs[in_names[0]], 0.0)
                outputs[name] = np.minimum(t, max_val) if max_val is not None else t
            elif cls == "ZeroPadding2D":
                (pt, pb), (pl, pr) = cfg["padding"]
                outputs[name] = np.pad(
                    outputs[in_names[0]], ((pt, pb), (pl, pr), (0, 0)), mode="constant"
                )
            elif cls == "Add":
                outputs[name] = outputs[in_names[0]] + outputs[in_names[1]]
            elif cls == "GlobalAveragePooling2D":
                outputs[name] = outputs[in_names[0]].mean(axis=(0, 1))
            elif cls == "Dropout":
                outputs[name] = outputs[in_names[0]]
            elif cls == "Dense":
                w = self.weights[name]
                t = outputs[in_names[0]] @ w["kernel"] + w["bias"]
                act = cfg.get("activation")
                if act == "relu":
                    t = np.maximum(t, 0.0)
                elif act == "sigmoid":
                    t = 1.0 / (1.0 + np.exp(-np.clip(t, -88.0, 88.0)))
                outputs[name] = t
            else:
                raise ValueError(f"Unsupported layer type: {cls}")

        return float(outputs[self.layers[-1]["name"]][0])

    def predict_image(self, image: Image.Image) -> dict:
        """Predict whether the image shows a person with or without a face mask."""
        arr = self.preprocess(image)
        raw_score = self.predict_raw(arr)

        # Class 0 ("mask") < 0.5, Class 1 ("no_mask") >= 0.5
        is_mask = raw_score < 0.5
        confidence = (1.0 - raw_score) if is_mask else raw_score
        label = "with_mask" if is_mask else "without_mask"
        display_label = "With Mask" if is_mask else "Without Mask"

        return {
            "label": label,
            "prediction": display_label,
            "is_mask": is_mask,
            "confidence": round(confidence, 6),
            "confidence_percent": round(confidence * 100, 2),
            "raw_score": round(raw_score, 6),
        }
