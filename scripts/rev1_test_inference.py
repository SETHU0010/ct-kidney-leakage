"""Revision analysis 1: re-run the seed-42 models on the grouped test split (CPU).

Answers reviewer comments M2 (direct RenalDualNet vs Swin-T test) and m8/M14 (full-split
accuracy of the FP32 and both INT8 students). Uses the model definitions and evaluation
transform of scripts/external_validation.py, which are copies of notebook Phase 5.1.

Reads : tables/manifest_final.csv, checkpoints/RenalDualNet_s42.pt,
        checkpoints/baseline_swin_tiny_patch4_window7_224.pt, deploy/*.onnx,
        the CT-KIDNEY images (verified against the manifest MD5s)
Writes: revision/test_logits_seed42.npz, revision/rev1_results.json
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import external_validation as ev  # noqa: E402  (model classes, eval_tf)

DATA = Path.home() / "Downloads/archive/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
OUT = ROOT / "revision"
K = 4
rng = np.random.default_rng(42)

man = pd.read_csv(ROOT / "tables/manifest_final.csv", usecols=["filename", "class_name", "label", "group_id", "split"])
test = man[man.split == "test"].reset_index(drop=True)
paths = [str(DATA / c / f) for c, f in zip(test.class_name, test.filename)]
y, g = test.label.values, test.group_id.values
print(f"test images {len(test)}, groups {test.group_id.nunique()}")


def macro_f1(yt, yp):
    cm = np.bincount(yt * K + yp, minlength=K * K).reshape(K, K).astype(float)
    tp = np.diag(cm); p = cm.sum(0); r = cm.sum(1)
    prec = np.divide(tp, p, out=np.zeros(K), where=p > 0)
    rec = np.divide(tp, r, out=np.zeros(K), where=r > 0)
    f = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    return f.mean()


def batched_torch(model, bs=32):
    model.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(paths), bs):
            x = torch.stack([ev.eval_tf(Image.open(p).convert("RGB")) for p in paths[s:s + bs]])
            out.append(model(x).numpy())
            print(f"  {min(s + bs, len(paths))}/{len(paths)}", end="\r", flush=True)
    print()
    return np.concatenate(out)


def batched_onnx(path, bs=32):
    so = ort.SessionOptions(); so.intra_op_num_threads = 8
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    nm = sess.get_inputs()[0].name; out = []
    for s in range(0, len(paths), bs):
        x = np.stack([ev.eval_tf(Image.open(p).convert("RGB")).numpy() for p in paths[s:s + bs]])
        out.append(sess.run(None, {nm: x})[0])
    return np.concatenate(out)


torch.set_num_threads(8)
logits = {}
t0 = time.time()
rd = ev.RenalDualNet()
rd.load_state_dict(torch.load(ROOT / "checkpoints/RenalDualNet_s42.pt", map_location="cpu", weights_only=False)["state_dict"])
print("RenalDualNet seed 42"); logits["RenalDualNet"] = batched_torch(rd); del rd
sw = ev.TimmBaseline("swin_tiny_patch4_window7_224.ms_in1k")
sw.load_state_dict(torch.load(ROOT / "checkpoints/baseline_swin_tiny_patch4_window7_224.pt", map_location="cpu", weights_only=False)["state_dict"])
print("Swin-T seed 42"); logits["SwinT"] = batched_torch(sw); del sw
for tag, f in [("student_fp32", "student_fp32.onnx"), ("int8_per_channel", "student_int8_per-channel_MinMax.onnx"),
               ("int8_per_tensor", "student_int8_per-tensor_MinMax.onnx"), ("int8_shipped", "student_int8.onnx")]:
    print(tag); logits[tag] = batched_onnx(ROOT / "deploy" / f)
print(f"inference done in {time.time() - t0:.0f}s")
np.savez_compressed(OUT / "test_logits_seed42.npz", y=y, g=g, **logits)

res = {"n_test": int(len(y)), "n_groups": int(len(np.unique(g))), "models": {}}
pred = {k: v.argmax(1) for k, v in logits.items()}
for k, p in pred.items():
    res["models"][k] = {"accuracy": float((p == y).mean()), "macro_f1": float(macro_f1(y, p))}
for k in ("student_fp32", "int8_per_channel", "int8_per_tensor", "int8_shipped"):
    res["models"][k]["agree_with_fp32"] = float((pred[k] == pred["student_fp32"]).mean())

# ---- paired comparison RenalDualNet vs Swin-T on identical test images (seed 42) ----
A, B = pred["RenalDualNet"], pred["SwinT"]
ug = np.unique(g); gi = np.searchsorted(ug, g)
obs_acc = (A == y).mean() - (B == y).mean()
obs_f1 = macro_f1(y, A) - macro_f1(y, B)
NPERM = 10000; ka = kf = 0
for _ in range(NPERM):
    flip = rng.random(len(ug)) < 0.5
    sw_ = flip[gi]
    pa, pb = np.where(sw_, B, A), np.where(sw_, A, B)
    if abs((pa == y).mean() - (pb == y).mean()) >= abs(obs_acc) - 1e-12: ka += 1
    if abs(macro_f1(y, pa) - macro_f1(y, pb)) >= abs(obs_f1) - 1e-12: kf += 1
boot_acc, boot_f1 = [], []
idx_by_g = [np.where(gi == j)[0] for j in range(len(ug))]
for _ in range(2000):
    pick = np.concatenate([idx_by_g[j] for j in rng.integers(0, len(ug), len(ug))])
    boot_acc.append((A[pick] == y[pick]).mean() - (B[pick] == y[pick]).mean())
    boot_f1.append(macro_f1(y[pick], A[pick]) - macro_f1(y[pick], B[pick]))
b01 = int(((A == y) & (B != y)).sum()); b10 = int(((A != y) & (B == y)).sum())
from scipy.stats import binomtest
mcn = binomtest(min(b01, b10), b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
res["paired_RenalDualNet_vs_SwinT"] = {
    "seed": 42, "acc_diff": float(obs_acc), "f1_diff": float(obs_f1),
    "acc_diff_ci": [float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5))],
    "f1_diff_ci": [float(np.percentile(boot_f1, 2.5)), float(np.percentile(boot_f1, 97.5))],
    "perm_p_acc": (1 + ka) / (NPERM + 1), "perm_p_f1": (1 + kf) / (NPERM + 1),
    "image_level_discordant": {"RDN_right_Swin_wrong": b01, "RDN_wrong_Swin_right": b10},
    "mcnemar_exact_p_image_level": float(mcn),
    "note": "group-level sign-flip permutation (10,000) and 2,000 group bootstrap; McNemar treats images as independent and is shown for reference only"}
(OUT / "rev1_results.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
