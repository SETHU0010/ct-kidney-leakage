"""Computer-vision helpers for the kidney CT app: Grad-CAM from ONNX, body masking,
contrast enhancement, edges, and overlays.

Grad-CAM without PyTorch
------------------------
The student is ResNet18: layer4 -> GlobalAveragePool -> Flatten -> Gemm (dropout is gone
at export). For logits y_c = sum_k W[c,k] * mean_ij A_k(i,j) + b_c, the Grad-CAM gradient
dy_c/dA_k(i,j) = W[c,k] / (H*W) is the same at every position, so the Grad-CAM channel
weights are W[c,k] / (H*W) and

    Grad-CAM_c = ReLU( sum_k W[c,k] * A_k ) / (H*W)

exactly. So we only need the layer4 feature map, which we expose as an extra ONNX output,
and the head weights, which we read from the graph (dequantizing them for INT8 models).
"""
import numpy as np
import cv2
import onnx
from onnx import numpy_helper, helper, TensorProto
import onnxruntime as ort

COLORMAPS = {"Jet": cv2.COLORMAP_JET, "Turbo": cv2.COLORMAP_TURBO,
             "Inferno": cv2.COLORMAP_INFERNO, "Viridis": cv2.COLORMAP_VIRIDIS}


# ------------------------------------------------------------------ ONNX surgery
class CamModel:
    """ONNX session returning (logits, layer4 features), plus the linear head."""

    def __init__(self, path, threads=4):
        m = onnx.load(str(path))
        g = m.graph
        producers = {o: n for n in g.node for o in n.output}
        inits = {t.name: t for t in g.initializer}

        gap = [n for n in g.node if n.op_type == "GlobalAveragePool"]
        if len(gap) != 1:
            raise ValueError(f"expected one GlobalAveragePool, found {len(gap)}")
        self.feat_name = gap[0].input[0]

        gemm = [n for n in g.node if n.op_type == "Gemm"]
        if len(gemm) != 1:
            raise ValueError(f"expected one Gemm head, found {len(gemm)}")
        gemm = gemm[0]
        W = self._const(gemm.input[1], producers, inits)
        b = self._const(gemm.input[2], producers, inits) if len(gemm.input) > 2 else 0.0
        trans_b = next((a.i for a in gemm.attribute if a.name == "transB"), 0)
        self.W = (W if trans_b else W.T).astype(np.float32)           # (classes, channels)
        self.b = np.broadcast_to(np.asarray(b, np.float32), (self.W.shape[0],)).copy()

        if self.feat_name not in [o.name for o in g.output]:
            g.output.append(helper.make_tensor_value_info(self.feat_name, TensorProto.FLOAT, None))
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(m.SerializeToString(), so,
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.logit_name = g.output[0].name

    @staticmethod
    def _const(name, producers, inits):
        """Value of a constant tensor, following one DequantizeLinear for QDQ models."""
        if name in inits:
            return numpy_helper.to_array(inits[name]).astype(np.float32)
        n = producers.get(name)
        if n is not None and n.op_type == "DequantizeLinear":
            q = numpy_helper.to_array(inits[n.input[0]]).astype(np.float32)
            s = numpy_helper.to_array(inits[n.input[1]]).astype(np.float32)
            zp = (numpy_helper.to_array(inits[n.input[2]]).astype(np.float32)
                  if len(n.input) > 2 and n.input[2] else np.zeros_like(s))
            axis = next((a.i for a in n.attribute if a.name == "axis"), 1)
            if s.ndim == 1 and s.size > 1:                      # per-channel
                shape = [1] * q.ndim; shape[axis] = -1
                s, zp = s.reshape(shape), zp.reshape(shape)
            return (q - zp) * s
        raise ValueError(f"cannot resolve constant {name!r}")

    def run(self, x):
        logits, feats = self.sess.run([self.logit_name, self.feat_name], {self.input_name: x})
        return logits[0], feats[0]                                  # (C,), (K,h,w)

    def gradcam(self, feats, class_idx):
        """Grad-CAM for one class on the layer4 grid (h, w), min-max normalised to [0,1]."""
        cam = np.maximum(np.tensordot(self.W[class_idx], feats, axes=(0, 0)), 0.0)
        return cam / cam.max() if cam.max() > 0 else cam

    def head_consistency(self, logits, feats):
        """max |W.mean(A)+b - logits|: confirms the closed-form Grad-CAM matches this graph."""
        return float(np.abs(self.W @ feats.mean((1, 2)) + self.b - logits).max())


# ------------------------------------------------------------------ image analysis
def to_gray_u8(rgb):
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)


def body_mask(gray, dark=0.10):
    """Patient body = largest bright connected component, holes filled.
    `dark` matches the 0.10 near-black cut-off used in the notebook's Phase 16 audit."""
    m = (gray >= dark * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros_like(gray, bool)
    body = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cnts, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(body)
    cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def clahe(gray, clip=2.0, tiles=8):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(gray)


def edges(gray, lo=40, hi=120):
    return cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), lo, hi)


def draw_contour(rgb, mask, color=(0, 255, 0), thickness=2):
    out = np.ascontiguousarray(rgb).copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, thickness)
    return out


def upsample(cam, size_wh):
    """Model-grid CAM -> image resolution. The model saw the image squashed to a square,
    so a per-axis resize maps the grid straight back onto the original geometry."""
    return np.clip(cv2.resize(cam.astype(np.float32), size_wh, interpolation=cv2.INTER_CUBIC), 0, 1)


def overlay(rgb, cam_full, alpha=0.45, cmap="Jet", mask=None):
    heat = cv2.applyColorMap((cam_full * 255).astype(np.uint8), COLORMAPS[cmap])
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    a = np.full(cam_full.shape, alpha, np.float32)
    if mask is not None:
        a = np.where(mask, a, 0.0)
    a = a[..., None]
    return (np.asarray(rgb, np.float32) * (1 - a) + heat * a).astype(np.uint8)


def attention_stats(cam_full, mask):
    """Share of Grad-CAM mass inside the body vs the share a uniform map would give.
    Ratio > 1: attention concentrated on anatomy. Ratio <= 1: no better than uniform,
    i.e. a possible background / border shortcut."""
    tot = float(cam_full.sum())
    area = float(mask.mean())
    inside = float(cam_full[mask].sum() / tot) if tot > 0 else float("nan")
    return {"cam_in_body": inside, "body_area": area,
            "concentration": inside / area if area > 0 else float("nan")}


def peak_box(cam_full, frac=0.5):
    """Bounding box (x, y, w, h) of the largest region with CAM >= frac * max."""
    m = (cam_full >= frac * cam_full.max()).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return tuple(int(v) for v in stats[i, :4])
