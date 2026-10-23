"""Revision analysis 2: checks that need no GPU.

(a) Injected-duplicate validation of the DAP grouping (reviewer M9, Supplement S8.2):
    perturbed copies of known images are hashed with the notebook's dHash/pHash and tested
    against the AND rule at (4, 7); MD5-only matching is the comparison.
(b) Independent near-duplicate audit of the grouped split (S8.3): a different image
    representation (32x32 grey thumbnails, Pearson correlation) is used to look for test
    images whose nearest training image is as close as two slices of the same scan.
(c) Control experiment (M9/M10): a 1-nearest-neighbour classifier and a linear classifier
    on raw thumbnails, scored by 5-fold random CV and 5-fold grouped CV, 3 repeats each.
(d) External cohorts (M4): where each true class goes, all four predicted classes.

Writes revision/rev2_*.csv and revision/rev2_results.json
"""
import io, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from scipy.fft import dct
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "Downloads/archive/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
OUT = ROOT / "revision"
rng = np.random.default_rng(42)
res = {}

man = pd.read_csv(ROOT / "tables/manifest_final.csv",
                  dtype={"dhash_hex": str, "phash_hex": str},
                  usecols=["filename", "class_name", "label", "group_id", "split", "dhash_hex", "phash_hex"])
paths = [DATA / c / f for c, f in zip(man.class_name, man.filename)]
bits = lambda s: np.frombuffer(s.encode(), dtype=np.uint8) - 48
DH = np.stack([bits(s) for s in man.dhash_hex]).astype(np.int32)
PH = np.stack([bits(s) for s in man.phash_hex]).astype(np.int32)


# ---- hashing identical to notebook Phase 2.1, but taking a PIL image ----
def _gray(img, size):
    return np.asarray(img.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32)

def dhash(img, n=8):
    a = _gray(img, n + 1)[:n, :n + 1]
    return (a[:, 1:] > a[:, :-1]).flatten().astype(np.int32)

def phash(img, n=8, hf=4):
    a = _gray(img, n * hf)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:n, :n]
    return (d > np.median(d[1:, 1:])).flatten().astype(np.int32)

def ham(a, B):
    return a.sum() + B.sum(1) - 2 * (B @ a)

# sanity: recomputed hashes must equal the stored manifest hashes
chk = rng.choice(len(man), 50, replace=False)
same = np.mean([(dhash(Image.open(paths[i])) == DH[i]).all() and (phash(Image.open(paths[i])) == PH[i]).all() for i in chk])
res["hash_reproduction_rate_on_50"] = float(same)
print("hash reproduction", same)


# ---- (a) injected duplicates ----
def jpeg(img, q):
    b = io.BytesIO(); img.convert("RGB").save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")

def shift(img, frac):
    w, h = img.size; dx, dy = int(w * frac), int(h * frac)
    return img.transform(img.size, Image.Transform.AFFINE, (1, 0, -dx, 0, 1, -dy))

def crop(img, frac):
    w, h = img.size; dx, dy = int(w * frac), int(h * frac)
    return img.crop((dx, dy, w - dx, h - dy)).resize((w, h), Image.BICUBIC)

PERTURB = {
    "exact copy": lambda im: im,
    "JPEG re-encode (q=75)": lambda im: jpeg(im, 75),
    "JPEG re-encode (q=50)": lambda im: jpeg(im, 50),
    "resize to 50% and back": lambda im: im.resize((im.size[0] // 2, im.size[1] // 2), Image.BICUBIC).resize(im.size, Image.BICUBIC),
    "brightness +10%": lambda im: ImageEnhance.Brightness(im).enhance(1.10),
    "contrast +10%": lambda im: ImageEnhance.Contrast(im).enhance(1.10),
    "Gaussian noise (sd 3)": lambda im: Image.fromarray(np.clip(np.asarray(im, np.float32) + rng.normal(0, 3, np.asarray(im).shape), 0, 255).astype(np.uint8)),
    "shift 2%": lambda im: shift(im, 0.02),
    "crop 5% border, resize back": lambda im: crop(im, 0.05),
    "horizontal flip": lambda im: im.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
}
src = np.concatenate([rng.choice(np.where(man.label == c)[0], 100, replace=False) for c in range(4)])
rows = []
t0 = time.time()
for i in src:
    img = Image.open(paths[i]).convert("RGB")
    gid = man.group_id.iloc[i]
    for name, fn in PERTURB.items():
        cp = fn(img)
        dh, ph = ham(dhash(cp), DH), ham(phash(cp), PH)
        link = (dh <= 4) & (ph <= 7)
        link_or = (dh <= 4) | (ph <= 7)
        rows.append({"perturbation": name, "src": i,
                     "and_links_source": bool(link[i]),
                     "and_links_source_group": bool(link[man.group_id.values == gid].any()),
                     "and_links_other_group": bool(link[man.group_id.values != gid].any()),
                     "or_links_source_group": bool(link_or[man.group_id.values == gid].any()),
                     "or_links_other_group": bool(link_or[man.group_id.values != gid].any()),
                     "md5_match": name == "exact copy",
                     "dhash_dist": int(dh[i]), "phash_dist": int(ph[i])})
inj = pd.DataFrame(rows)
summ = inj.groupby("perturbation", sort=False).agg(
    n=("src", "size"),
    md5_recall=("md5_match", "mean"),
    and_recall_source=("and_links_source", "mean"),
    and_recall_group=("and_links_source_group", "mean"),
    and_false_link_other_group=("and_links_other_group", "mean"),
    or_recall_group=("or_links_source_group", "mean"),
    or_false_link_other_group=("or_links_other_group", "mean"),
    median_dhash=("dhash_dist", "median"), median_phash=("phash_dist", "median")).reset_index()
summ.to_csv(OUT / "rev2_injected_duplicates.csv", index=False)
print(summ.to_string()); print(f"injected in {time.time() - t0:.0f}s")
res["injected_duplicates"] = summ.to_dict(orient="records")


# ---- thumbnails for (b) and (c) ----
t0 = time.time()
TH = np.stack([np.asarray(Image.open(p).convert("L").resize((32, 32), Image.LANCZOS), np.float32).ravel() for p in paths])
Z = TH - TH.mean(1, keepdims=True); Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
print(f"thumbnails in {time.time() - t0:.0f}s")
y, grp = man.label.values, man.group_id.values


# ---- (b) independent near-duplicate audit ----
def nn_corr(q_idx, ref_idx, exclude_self=False):
    out = np.empty(len(q_idx), np.float32)
    for s in range(0, len(q_idx), 512):
        C = Z[q_idx[s:s + 512]] @ Z[ref_idx].T
        if exclude_self:
            for r, qi in enumerate(q_idx[s:s + 512]):
                C[r, ref_idx == qi] = -1
        out[s:s + 512] = C.max(1)
    return out

# reference: how close is an image to the nearest other slice of its own (estimated) scan?
within = []
for gid in np.unique(grp):
    ix = np.where(grp == gid)[0]
    if len(ix) < 2: continue
    C = Z[ix] @ Z[ix].T; np.fill_diagonal(C, -1); within.append(C.max(1))
within = np.concatenate(within)
thr = float(np.percentile(within, 5))

tr, te = np.where(man.split == "train")[0], np.where(man.split == "test")[0]
grouped_nn = nn_corr(te, tr)
perm = rng.permutation(len(man)); ntr = int(0.7 * len(man)); nva = int(0.15 * len(man))
rtr, rte = perm[:ntr], perm[ntr + nva:]
random_nn = nn_corr(rte, rtr)
audit = {
    "representation": "32x32 grey thumbnail, Pearson correlation (independent of dHash/pHash)",
    "within_scan_nn_corr_median": float(np.median(within)),
    "threshold_5th_pct_within_scan": thr,
    "grouped_split_test_n": int(len(te)),
    "grouped_split_test_with_train_nn_above_thr": int((grouped_nn >= thr).sum()),
    "grouped_split_frac_above_thr": float((grouped_nn >= thr).mean()),
    "grouped_split_test_nn_corr_median": float(np.median(grouped_nn)),
    "random_split_test_n": int(len(rte)),
    "random_split_test_with_train_nn_above_thr": int((random_nn >= thr).sum()),
    "random_split_frac_above_thr": float((random_nn >= thr).mean()),
    "random_split_test_nn_corr_median": float(np.median(random_nn)),
}
pd.DataFrame({"grouped_test_nn_corr": pd.Series(grouped_nn), "random_test_nn_corr": pd.Series(random_nn)}).to_csv(OUT / "rev2_independent_audit_nn.csv", index=False)
res["independent_audit"] = audit
print(json.dumps(audit, indent=2))


# ---- (c) control experiment: memorisation under random vs grouped 5-fold CV ----
def one_nn_predict(tr_i, te_i):
    pred = np.empty(len(te_i), int)
    for s in range(0, len(te_i), 512):
        C = Z[te_i[s:s + 512]] @ Z[tr_i].T
        pred[s:s + 512] = y[tr_i][C.argmax(1)]
    return pred

ctrl = []
for rep in range(3):
    for scheme, cv in [("random 5-fold", StratifiedKFold(5, shuffle=True, random_state=rep)),
                       ("grouped 5-fold", StratifiedGroupKFold(5, shuffle=True, random_state=rep))]:
        for fold, (tr_i, te_i) in enumerate(cv.split(Z, y, grp)):
            p1 = one_nn_predict(tr_i, te_i)
            lr = make_pipeline(StandardScaler(), PCA(64, random_state=0), LogisticRegression(max_iter=3000, class_weight="balanced"))
            lr.fit(TH[tr_i], y[tr_i]); p2 = lr.predict(TH[te_i])
            ctrl.append({"repeat": rep, "scheme": scheme, "fold": fold,
                         "1nn_macro_f1": f1_score(y[te_i], p1, average="macro"), "1nn_acc": float((p1 == y[te_i]).mean()),
                         "linear_macro_f1": f1_score(y[te_i], p2, average="macro"), "linear_acc": float((p2 == y[te_i]).mean())})
    print("repeat", rep, "done")
ctrl = pd.DataFrame(ctrl); ctrl.to_csv(OUT / "rev2_control_cv_folds.csv", index=False)
cs = ctrl.groupby("scheme").agg(["mean", "std"])[["1nn_macro_f1", "1nn_acc", "linear_macro_f1", "linear_acc"]]
print(cs.to_string())
res["control_cv"] = {s: {c: {"mean": float(cs.loc[s, (c, "mean")]), "std": float(cs.loc[s, (c, "std")])}
                         for c in ["1nn_macro_f1", "1nn_acc", "linear_macro_f1", "linear_acc"]} for s in cs.index}


# ---- (d) external cohorts: true class x predicted class ----
ext = {}
rows = []
for coh in ["yildirim", "axial"]:
    d = pd.read_csv(ROOT / "external_validation" / f"{coh}_predictions.csv")
    for (m, c), gdf in d.groupby(["model", "class_name"]):
        vc = gdf.pred.value_counts(normalize=True)
        rows.append({"cohort": coh, "model": m, "true_class": c, "n": len(gdf),
                     **{f"pred_{k}": float(vc.get(k, 0.0)) for k in ["Cyst", "Normal", "Stone", "Tumor"]}})
extdf = pd.DataFrame(rows); extdf.to_csv(OUT / "rev2_external_confusion.csv", index=False)
print(extdf.round(3).to_string())
res["external_confusion"] = extdf.to_dict(orient="records")

(OUT / "rev2_results.json").write_text(json.dumps(res, indent=2))
print("saved")
