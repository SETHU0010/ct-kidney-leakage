"""Kidney CT triage - Streamlit front end for the distilled ResNet18 student.

Run from this folder:
    pip install -r requirements.txt
    streamlit run app.py

Everything model-specific (temperature, abstention threshold, conformal q_hat, test
metrics) is read from deploy_config.json, which Phase 18.1 of Obj1.ipynb computes on the
SHIPPED model's own validation logits. If that file is missing the app still runs, but
calibration, abstention and prediction sets are switched off, because values borrowed from
the teacher model do not transfer to the student.

Grad-CAM heatmaps and the image-analysis tools live in vision.py.
"""
import io, json, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import vision as V

HERE = Path(__file__).parent
CONFIG_FILE = HERE / "deploy_config.json"
IMG_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
GALLERY_MAX = 24
# every training image was an 8-bit RGB JPEG; these modes never occurred in training
HIGH_DEPTH_MODES = ("I;16", "I;16B", "I;16L", "I", "F")

DEFAULTS = {
    "classes": ["Cyst", "Normal", "Stone", "Tumor"],
    "img_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "model_file": "student_int8.onnx",
    "fallback_model_file": "student_fp32.onnx",
    "temperature": 1.0,
    "conf_threshold": None,
    "q_hat": None,
    "conformal_alpha": None,
    "target_coverage": None,
    "calibrated_on": None,
    "metrics": {},
}

st.set_page_config(page_title="Kidney CT Triage", page_icon="🩻", layout="wide")


# ---------------------------------------------------------------- config and model
def load_config():
    # read on every rerun (tiny file) so a freshly written config is picked up live
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        cfg.update(json.loads(CONFIG_FILE.read_text()))
        cfg["_calibrated"] = cfg.get("calibrated_on") is not None
    else:
        cfg["_calibrated"] = False
    return cfg


@st.cache_resource
def load_model(model_file):
    return V.CamModel(HERE / model_file)


def available_models(cfg):
    names = [cfg["model_file"], cfg.get("fallback_model_file")]
    return [n for n in dict.fromkeys(names) if n and (HERE / n).exists()]


# ---------------------------------------------------------------- inference
def to_rgb(img):
    """PIL image of any mode -> 8-bit RGB. 16-bit / float CT exports are min-max scaled
    first, because PIL's own convert('RGB') clips them to white. The model never saw such
    images (the training set is 8-bit JPEG only), so analyse() flags them as rescaled."""
    if img.mode in HIGH_DEPTH_MODES:
        a = np.asarray(img, dtype=np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(a)
        img = Image.fromarray(a.astype(np.uint8))
    return img.convert("RGB")


def preprocess(rgb, cfg):
    """Mirrors eval_tf in the notebook: Resize((S,S)) -> ToTensor -> Normalize."""
    s = cfg["img_size"]
    a = np.asarray(rgb.resize((s, s), Image.BILINEAR), dtype=np.float32) / 255.0
    a = a.transpose(2, 0, 1)
    mean = np.array(cfg["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.array(cfg["std"], dtype=np.float32).reshape(3, 1, 1)
    return ((a - mean) / std)[None, ...].astype(np.float32)


def softmax(z, T=1.0):
    z = np.asarray(z, dtype=np.float64) / T
    e = np.exp(z - z.max())
    return e / e.sum()


def analyse(model, img, cfg, use_cal):
    """One forward pass -> probabilities, layer4 features, timing, and the RGB image."""
    rgb = to_rgb(img)
    t = time.perf_counter()
    logits, feats = model.run(preprocess(rgb, cfg))
    ms = (time.perf_counter() - t) * 1000
    T = cfg["temperature"] if (use_cal and cfg["_calibrated"]) else 1.0
    return {"rgb": rgb, "probs": softmax(logits, T), "feats": feats, "ms": ms,
            "rescaled": img.mode in HIGH_DEPTH_MODES, "mode": img.mode}


def prediction_set(probs, q_hat):
    """Split-conformal set {k : p_k >= 1 - q_hat}; the arg-max is always kept so the set
    is never empty (same convention as Phase 10.1)."""
    keep = probs >= (1.0 - q_hat)
    keep[int(probs.argmax())] = True
    return [int(i) for i in np.flatnonzero(keep)]


def decide(probs, cfg, use_abstain):
    top = int(probs.argmax())
    thr = cfg["conf_threshold"]
    abstain = bool(use_abstain and thr is not None and probs[top] < thr)
    return top, abstain


def heatmap(model, res, class_idx):
    """Grad-CAM at full image resolution plus body mask and attention statistics."""
    rgb = np.asarray(res["rgb"])
    cam = V.upsample(model.gradcam(res["feats"], class_idx), (rgb.shape[1], rgb.shape[0]))
    mask = V.body_mask(V.to_gray_u8(rgb))
    return cam, mask, V.attention_stats(cam, mask)


def png_bytes(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def concentration_label(c):
    if not np.isfinite(c):
        return "no body region found"
    if c >= 1.5:
        return "focused on anatomy"
    if c > 1.0:
        return "mildly anatomy-focused"
    return "no better than uniform: possible background/border shortcut"


# ---------------------------------------------------------------- UI
cfg = load_config()
CLASSES = cfg["classes"]
models = available_models(cfg)

st.title("Kidney CT triage")
st.caption("Research prototype for a benchmark-integrity study. **Not a medical device. "
           "Not for clinical use.**")

if not models:
    st.error(f"No ONNX model found next to app.py (looked for {cfg['model_file']}). "
             "Run Phase 17 of Obj1.ipynb first.")
    st.stop()

if not cfg["_calibrated"]:
    st.warning("deploy_config.json not found. Showing raw (uncalibrated) probabilities; "
               "abstention and prediction sets are disabled. Run Phase 18.1 of the notebook "
               "to calibrate the shipped model.")

with st.sidebar:
    st.header("Settings")
    model_file = st.selectbox("Model", models,
                              help="INT8 is the shipped model; FP32 is the unquantized reference.")
    use_cal = st.checkbox("Apply temperature calibration", value=cfg["_calibrated"],
                          disabled=not cfg["_calibrated"])
    use_abstain = st.checkbox("Abstain below confidence threshold",
                              value=cfg["conf_threshold"] is not None,
                              disabled=cfg["conf_threshold"] is None)
    show_set = st.checkbox("Show conformal prediction set",
                           value=cfg["q_hat"] is not None, disabled=cfg["q_hat"] is None)
    st.divider()
    st.subheader("Heatmap style")
    cmap = st.selectbox("Colormap", list(V.COLORMAPS), index=1)
    alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.45, 0.05)
    st.divider()
    st.markdown(f"**Model file:** `{model_file}`")
    if cfg["_calibrated"]:
        st.markdown(f"**Temperature:** {cfg['temperature']:.4f}")
    if cfg["conf_threshold"] is not None:
        st.markdown(f"**Abstention threshold:** {cfg['conf_threshold']:.4f} "
                    f"(target coverage {cfg['target_coverage']:.0%})")
    if cfg["q_hat"] is not None:
        st.markdown(f"**Conformal α:** {cfg['conformal_alpha']}  ·  q̂ = {cfg['q_hat']:.4f}")
    if model_file != cfg["model_file"] and cfg["_calibrated"]:
        st.info("Calibration values were fitted on the shipped model "
                f"({cfg['model_file']}); they are approximate for this one.")

model = load_model(model_file)
tab_single, tab_batch, tab_about = st.tabs(["Single image", "Batch", "About the model"])

# ================================================================ single image
with tab_single:
    up = st.file_uploader("Upload a kidney CT slice", type=IMG_TYPES, key="single")
    if up:
        img = Image.open(up)
        res = analyse(model, img, cfg, use_cal)
        probs = res["probs"]
        top, abstain = decide(probs, cfg, use_abstain)
        rgb = np.asarray(res["rgb"])
        top_cam, top_mask, _ = heatmap(model, res, top)
        if res["rescaled"]:
            st.warning(f"{res['mode']} image (more than 8 bits per pixel). It was min-max "
                       "rescaled to 8 bits, but every training image was an 8-bit JPEG, so "
                       "this input is out of distribution. The window/level the model saw is "
                       "not the one it was trained on; treat the prediction with caution.")

        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.image(V.overlay(rgb, top_cam, alpha, cmap),
                     caption=f"{up.name}  ·  {img.size[0]}×{img.size[1]}  ·  Grad-CAM for "
                             f"{CLASSES[top]}", width="stretch")
        with c2:
            if abstain:
                st.warning("### Abstain: refer to a radiologist")
                st.write(f"Top class **{CLASSES[top]}** at {probs[top]:.1%} is below the "
                         f"{cfg['conf_threshold']:.1%} threshold.")
            else:
                st.success(f"### {CLASSES[top]}")
                label = "Calibrated confidence" if (use_cal and cfg["_calibrated"]) else "Confidence"
                st.metric(label, f"{probs[top]:.1%}")
            if show_set and cfg["q_hat"] is not None:
                ps = prediction_set(probs, cfg["q_hat"])
                st.markdown(f"**Prediction set ({1 - cfg['conformal_alpha']:.0%} target "
                            f"coverage):** " + ", ".join(CLASSES[i] for i in ps))
            st.bar_chart(pd.DataFrame({"probability": probs}, index=CLASSES),
                         horizontal=True, height=200)
            st.caption(f"Inference + features {res['ms']:.1f} ms on CPU")

        t_cam, t_cv = st.tabs(["Grad-CAM heatmap", "Computer vision analysis"])

        # ---------------- Grad-CAM
        with t_cam:
            k1, k2, k3, k4 = st.columns([1.3, 1, 1, 1])
            cls = k1.selectbox("Explain class", CLASSES, index=top,
                               help="Grad-CAM for any class, not only the prediction: "
                                    "where would the model look for evidence of this class?")
            body_only = k2.checkbox("Restrict to body", value=False)
            outline = k3.checkbox("Body outline", value=True)
            box = k4.checkbox("Box around peak", value=True)

            ci = CLASSES.index(cls)
            cam, mask, stats = heatmap(model, res, ci)
            view = V.overlay(rgb, cam, alpha, cmap, mask if body_only else None)
            if outline:
                view = V.draw_contour(view, mask, (0, 255, 0), 2)
            pb = V.peak_box(cam) if box else None
            if pb:
                x, y, w, h = pb
                view = cv2.rectangle(np.ascontiguousarray(view), (x, y), (x + w, y + h),
                                     (255, 255, 255), 2)

            h1, h2 = st.columns(2)
            h1.image(view, caption=f"Grad-CAM overlay: {cls} (p = {probs[ci]:.1%})",
                     width="stretch")
            raw = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8),
                                                 V.COLORMAPS[cmap]), cv2.COLOR_BGR2RGB)
            h2.image(raw, caption="Raw heatmap (7×7 layer4 grid, upsampled)",
                     width="stretch")

            m1, m2, m3 = st.columns(3)
            m1.metric("Grad-CAM mass inside body", f"{stats['cam_in_body']:.1%}")
            m2.metric("Body area (uniform baseline)", f"{stats['body_area']:.1%}")
            m3.metric("Concentration ratio", f"{stats['concentration']:.2f}×",
                      help="Mass inside body ÷ body area. 1.0 = what a flat heatmap would "
                           "score. The notebook's Phase 16 audit had no such baseline.")
            st.caption(f"Reading: {concentration_label(stats['concentration'])}.")
            st.download_button("Download overlay (PNG)", png_bytes(view),
                               file_name=f"{Path(up.name).stem}_gradcam_{cls}.png",
                               mime="image/png")

            with st.expander("Heatmaps for every class"):
                cols = st.columns(len(CLASSES))
                for i, c in enumerate(CLASSES):
                    ci_cam, _, ci_st = heatmap(model, res, i)
                    cols[i].image(V.overlay(rgb, ci_cam, alpha, cmap), width="stretch",
                                  caption=f"{c}: p={probs[i]:.1%} · "
                                          f"conc {ci_st['concentration']:.2f}×")

            st.caption("How this is computed: the student is ResNet18 (layer4 → global average "
                       "pool → linear head), so the Grad-CAM gradient is the same at every "
                       "position and Grad-CAM = ReLU(W_c · A) exactly. It is computed from an "
                       "extra ONNX output, with no PyTorch needed, and matches autograd "
                       "Grad-CAM to about 1e-6.")

        # ---------------- computer vision
        with t_cv:
            st.caption("Display-only analysis. The model always sees the original image. "
                       "Enhanced views are for the reader, not for inference.")
            gray = V.to_gray_u8(rgb)
            s1, s2, s3 = st.columns(3)
            clip = s1.slider("CLAHE clip limit", 1.0, 6.0, 2.0, 0.5)
            lo, hi = s2.slider("Canny thresholds", 0, 255, (40, 120))
            dark = s3.slider("Body threshold (fraction of 255)", 0.02, 0.40, 0.10, 0.01)
            cv_mask = V.body_mask(gray, dark)

            v1, v2, v3 = st.columns(3)
            v1.image(V.clahe(gray, clip), caption="CLAHE contrast enhancement",
                     width="stretch", clamp=True)
            v2.image(V.draw_contour(rgb, cv_mask, (0, 255, 0), 3),
                     caption=f"Body segmentation ({cv_mask.mean():.1%} of image)",
                     width="stretch")
            v3.image(V.edges(gray, lo, hi), caption="Canny edges", width="stretch",
                     clamp=True)

            g1, g2 = st.columns([1.4, 1])
            with g1:
                hist = np.bincount(gray.ravel(), minlength=256)
                st.markdown("**Intensity histogram**")
                st.area_chart(pd.DataFrame({"pixels": hist}), height=220)
            with g2:
                body_px = gray[cv_mask] if cv_mask.any() else gray.ravel()
                st.markdown("**Image properties**")
                st.dataframe(pd.DataFrame({
                    "property": ["Size (W×H)", "Aspect ratio", "Source mode", "Mean intensity",
                                 "Std intensity", "Body mean intensity", "Near-black pixels",
                                 "Body area"],
                    "value": [f"{img.size[0]}×{img.size[1]}", f"{img.size[0] / img.size[1]:.3f}",
                              img.mode, f"{gray.mean():.1f}", f"{gray.std():.1f}",
                              f"{body_px.mean():.1f}", f"{(gray < dark * 255).mean():.1%}",
                              f"{cv_mask.mean():.1%}"]}),
                    hide_index=True, width="stretch")
                if img.size != (512, 512):
                    st.info("Non-512×512 input. In this dataset every such image is a coronal "
                            "reformat, and each scan was exported at its own size, so size "
                            "identifies the scan, not the class (Phase 2.3b). The model squashes "
                            f"every image to {cfg['img_size']}×{cfg['img_size']}, so aspect ratio "
                            "is distorted; that is expected, not an error.")

# ================================================================ batch
with tab_batch:
    ups = st.file_uploader("Upload several slices", type=IMG_TYPES,
                           accept_multiple_files=True, key="batch")
    gallery = st.checkbox(f"Show Grad-CAM gallery (first {GALLERY_MAX})", value=True)
    if ups:
        rows, thumbs, prog = [], [], st.progress(0.0)
        for k, f in enumerate(ups):
            try:
                res = analyse(model, Image.open(f), cfg, use_cal)
            except Exception as e:
                rows.append({"file": f.name, "prediction": f"ERROR: {type(e).__name__}"})
                continue
            probs = res["probs"]
            top, abstain = decide(probs, cfg, use_abstain)
            cam, mask, stt = heatmap(model, res, top)
            row = {"file": f.name,
                   "prediction": "ABSTAIN" if abstain else CLASSES[top],
                   "top_class": CLASSES[top],
                   "confidence": round(float(probs[top]), 4),
                   "cam_in_body": round(stt["cam_in_body"], 4),
                   "concentration": round(stt["concentration"], 3),
                   "note": f"{res['mode']} rescaled: out of distribution" if res["rescaled"] else ""}
            if show_set and cfg["q_hat"] is not None:
                row["prediction_set"] = " | ".join(CLASSES[i] for i in prediction_set(probs, cfg["q_hat"]))
            row.update({c: round(float(probs[i]), 4) for i, c in enumerate(CLASSES)})
            rows.append(row)
            if gallery and len(thumbs) < GALLERY_MAX:
                thumbs.append((V.overlay(np.asarray(res["rgb"]), cam, alpha, cmap),
                               f"{f.name} · {row['prediction']} {probs[top]:.0%} · "
                               f"{stt['concentration']:.2f}×"))
            prog.progress((k + 1) / len(ups))
        df = pd.DataFrame(rows)
        n_abs = int((df["prediction"] == "ABSTAIN").sum())
        n_err = int(df["prediction"].str.startswith("ERROR").sum())
        n_short = int((df.get("concentration", pd.Series(dtype=float)) <= 1.0).sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Classified", len(df) - n_abs - n_err)
        m2.metric("Referred for review", n_abs)
        m3.metric("Unreadable", n_err)
        m4.metric("Heatmap not on anatomy", n_short,
                  help="Concentration ratio ≤ 1: Grad-CAM mass inside the body is no higher "
                       "than a uniform map would give.")
        st.dataframe(df, width="stretch", hide_index=True)
        buf = io.StringIO(); df.to_csv(buf, index=False)
        st.download_button("Download results (CSV)", buf.getvalue(),
                           file_name="kidney_ct_predictions.csv", mime="text/csv")
        if thumbs:
            st.markdown("**Grad-CAM gallery** (heatmap for the predicted class)")
            cols = st.columns(4)
            for i, (im, cap) in enumerate(thumbs):
                cols[i % 4].image(im, caption=cap, width="stretch")

# ================================================================ about
with tab_about:
    m = cfg.get("metrics", {})
    st.subheader("What this model is")
    st.markdown(
        f"- **Architecture:** {cfg.get('student_backbone', 'ResNet18')} student, distilled "
        "from a ConvNeXt-T + Swin-T teacher, exported to ONNX and quantized to INT8.\n"
        "- **Data split:** Duplicate-Aware Partitioning. Near-duplicate CT slices were grouped "
        "by dual perceptual hashing (dHash AND pHash) and each group was kept inside one "
        "split. Under a random split 99.8% of test images have a near-duplicate in "
        "training, which is why published accuracies on this benchmark are near 100%.\n"
        "- **Explanations:** Grad-CAM on the last ResNet block (7×7 grid), computed in closed "
        "form from the ONNX graph. The concentration ratio compares heatmap mass inside the "
        "segmented body with the body's share of the image.")
    if m:
        st.subheader("Measured performance of the shipped model (DAP test split)")
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{m['test_accuracy']:.3f}")
        cols[1].metric("Macro-F1", f"{m['test_macro_f1']:.3f}")
        if m.get("selective_coverage") is not None:
            cols[2].metric("Coverage at threshold", f"{m['selective_coverage']:.1%}")
            cols[3].metric("Accuracy when answering", f"{m['selective_accuracy']:.3f}")
        if m.get("per_class"):
            st.dataframe(pd.DataFrame(m["per_class"]).round(4), hide_index=True)
        st.caption(f"{m.get('n_test', '?')} test images from {m.get('n_test_groups', '?')} "
                   "scan groups. Single dataset, single seed; the multi-seed spread is "
                   "reported in the paper.")
    st.subheader("Limitations")
    st.markdown(
        "- One public dataset (single source); no external cohort and no reader study.\n"
        "- Operates on single 2-D slices, not whole studies.\n"
        "- Confidence and prediction sets assume the input looks like the training data. "
        "Images from other scanners, windows or crops are out of distribution.\n"
        "- Grad-CAM is coarse (7×7) and shows where evidence was found, not that the "
        "reasoning is clinically valid.")
