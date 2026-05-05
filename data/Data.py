# MATH 345 FINAL PROJECT
# Sai Manvith Chatrathi, Andy Le, Qasim Masood, Giorgi Karazanashvili, Erika Severino

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.datasets import load_breast_cancer

os.makedirs("outputs", exist_ok=True)

# NOTE: sklearn's load_breast_cancer() is the exact same UCI dataset (id=17).
# We use it here since ucimlrepo requires an external network connection.

# 1. LOAD DATASET
print("Data Loading, Preprocessing & Histograms")

raw  = load_breast_cancer()
X    = pd.DataFrame(raw.data, columns=raw.feature_names)

# sklearn encodes: 0 = malignant, 1 = benign  →  remap so M=1, B=0 (our convention)
y    = pd.Series(raw.target).map({0: 1, 1: 0}).rename("diagnosis")

df   = pd.concat([X, y], axis=1)

print(f"\n[1] Dataset loaded successfully.")
print(f"    Shape  : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"    Source : Breast Cancer Wisconsin (Diagnostic) — UCI id=17\n")

# 2. RENAME MEAN FEATURES FOR CLARITY
rename_map = {
    "mean radius"            : "radius_mean",
    "mean texture"           : "texture_mean",
    "mean perimeter"         : "perimeter_mean",
    "mean area"              : "area_mean",
    "mean smoothness"        : "smoothness_mean",
    "mean compactness"       : "compactness_mean",
    "mean concavity"         : "concavity_mean",
    "mean concave points"    : "concave_points_mean",
    "mean symmetry"          : "symmetry_mean",
    "mean fractal dimension" : "fractal_dimension_mean",
}
df.rename(columns=rename_map, inplace=True)

MEAN_FEATURES = list(rename_map.values())

# 3. DATA QUALITY CHECKS
print("[2] Diagnosis Distribution")
counts  = df["diagnosis"].value_counts().sort_index(ascending=False)
n_total = len(df)
n_M     = int(counts[1])
n_B     = int(counts[0])

print(f"    Benign    (B / 0) : {n_B:>4}  ({n_B/n_total*100:.1f}%)")
print(f"    Malignant (M / 1) : {n_M:>4}  ({n_M/n_total*100:.1f}%)")
print(f"    Empirical prior   :  P(Malignant) = {n_M}/{n_total} ≈ {n_M/n_total:.4f}")

print(f"\n[3] Missing Value Check")
print("-" * 62)
missing = df[MEAN_FEATURES].isnull().sum()
if missing.sum() == 0:
    print("No missing values detected across all 10 mean features.")
else:
    print(missing[missing > 0])

print(f"\n[4] First 5 rows (mean features + diagnosis):")
print(df[MEAN_FEATURES + ["diagnosis"]].head().to_string(index=True))

# 4. SUMMARY STATISTICS BY GROUP
benign    = df[df["diagnosis"] == 0]
malignant = df[df["diagnosis"] == 1]

print(f"\n[5] Summary Statistics — Mean ± Std by Diagnosis Group")
print("-" * 62)
print(f"  {'Feature':<28} {'Benign μ (σ)':<22} {'Malignant μ (σ)':<22} Cohen's d")
print(f"  {'-'*25:<28} {'-'*18:<22} {'-'*18:<22} {'-'*8}")

sep_scores = {}
for feat in MEAN_FEATURES:
    bmu, bsd = benign[feat].mean(),    benign[feat].std()
    mmu, msd = malignant[feat].mean(), malignant[feat].std()
    pooled_sd = np.sqrt((bsd**2 + msd**2) / 2)
    d = abs(mmu - bmu) / pooled_sd
    sep_scores[feat] = d
    print(f"  {feat:<28} {bmu:7.4f}  ({bsd:.4f})     "
          f"{mmu:7.4f}  ({msd:.4f})    {d:.3f}")

ranked = sorted(sep_scores.items(), key=lambda x: x[1], reverse=True)
print(f"\n  Feature Separation Ranking (Cohen's d):")
for rank, (feat, d) in enumerate(ranked, 1):
    bar   = "█" * int(d * 5)
    label = "★ large" if d > 1.5 else ("▲ mod." if d > 0.8 else "▼ small")
    print(f"  {rank:2}. {feat:<28} d={d:.3f}  {bar:<14} {label}")

# 5. FIGURE 1 — Overview: all 10 features
BLUE = "#4C8DF5"
RED  = "#E05252"
BG   = "#F4F6F9"

print(f"\n[6] Generating figures ...")

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "Figure 1: Histograms of All 10 Mean Features: Benign vs. Malignant",
    fontsize=13, fontweight="bold", y=.98
)

for ax, feat in zip(axes.flat, MEAN_FEATURES):
    b_vals = benign[feat]
    m_vals = malignant[feat]
    bins   = np.linspace(min(b_vals.min(), m_vals.min()),
                         max(b_vals.max(), m_vals.max()), 30)

    ax.hist(b_vals, bins=bins, alpha=0.6, color=BLUE,
            density=True, edgecolor="white", linewidth=0.4, label="Benign")
    ax.hist(m_vals, bins=bins, alpha=0.6, color=RED,
            density=True, edgecolor="white", linewidth=0.4, label="Malignant")

    # Fitted Normal curves
    x_range = np.linspace(bins[0], bins[-1], 200)
    ax.plot(x_range, stats.norm.pdf(x_range, b_vals.mean(), b_vals.std()),
            color=BLUE, lw=1.8, linestyle="--")
    ax.plot(x_range, stats.norm.pdf(x_range, m_vals.mean(), m_vals.std()),
            color=RED,  lw=1.8, linestyle="--")

    title = feat.replace("_mean", "").replace("_", " ").title()
    d     = sep_scores[feat]
    tag   = "★" if d > 1.5 else ("▲" if d > 0.8 else "▼")
    ax.set_title(f"{title}  {tag} d={d:.2f}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Value", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_facecolor("#EAECF0")
    ax.grid(True, color="white", linewidth=0.6)

handles = [
    plt.Rectangle((0,0),1,1, color=BLUE, alpha=0.7, label="Benign (B)"),
    plt.Rectangle((0,0),1,1, color=RED,  alpha=0.7, label="Malignant (M)"),
    plt.Line2D([0],[0], color="gray", lw=1.8, linestyle="--", label="Fitted Normal PDF"),
]
fig.legend(handles=handles, loc="upper right", fontsize=10,
           ncol=3, bbox_to_anchor=(1.0, 0.98))
plt.tight_layout()
plt.savefig("outputs/fig1_all_features.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("    Saved → outputs/fig1_all_features.png")

# 6. FIGURE 2 — Deep-dive: top 5 features + Bayesian posterior preview
top5    = [f for f, _ in ranked[:5]]
prior_M = n_M / n_total
prior_B = 1 - prior_M

fig = plt.figure(figsize=(18, 9))
fig.patch.set_facecolor(BG)
fig.suptitle(
    "Figure 2: Top 5 Features: Overlaid Normal PDFs (top) & Bayesian Posterior Preview (bottom)",
    fontsize=12, fontweight="bold", y=.98
)
gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.45, wspace=0.3)

for col, feat in enumerate(top5):
    b_vals = benign[feat]
    m_vals = malignant[feat]
    bmu, bsd = b_vals.mean(), b_vals.std()
    mmu, msd = m_vals.mean(), m_vals.std()
    bins     = np.linspace(min(b_vals.min(), m_vals.min()),
                           max(b_vals.max(), m_vals.max()), 35)
    x_range  = np.linspace(bins[0], bins[-1], 400)

    # ── Top row: histogram + Normal PDF ──────────────────────────────────────
    ax_top = fig.add_subplot(gs[0, col])
    ax_top.hist(b_vals, bins=bins, alpha=0.45, color=BLUE, density=True,
                edgecolor="white", linewidth=0.4)
    ax_top.hist(m_vals, bins=bins, alpha=0.45, color=RED,  density=True,
                edgecolor="white", linewidth=0.4)
    ax_top.plot(x_range, stats.norm.pdf(x_range, bmu, bsd),
                color=BLUE, lw=2.2, linestyle="--")
    ax_top.plot(x_range, stats.norm.pdf(x_range, mmu, msd),
                color=RED,  lw=2.2, linestyle="--")
    ax_top.axvline(bmu, color=BLUE, lw=1, linestyle=":", alpha=0.8)
    ax_top.axvline(mmu, color=RED,  lw=1, linestyle=":", alpha=0.8)

    title = feat.replace("_mean","").replace("_"," ").title()
    d     = sep_scores[feat]
    ax_top.set_title(f"{title}\nd = {d:.2f}", fontsize=9, fontweight="bold")
    ax_top.set_ylabel("Density", fontsize=8)
    ax_top.tick_params(labelsize=7)
    ax_top.set_facecolor("#EAECF0")
    ax_top.grid(True, color="white", linewidth=0.6)
    ax_top.text(0.03, 0.97,
                f"B: μ={bmu:.2f}, σ={bsd:.2f}\nM: μ={mmu:.2f}, σ={msd:.2f}",
                transform=ax_top.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8))

    # ── Bottom row: Bayesian posterior P(M | x) ──────────────────────────────
    ax_bot = fig.add_subplot(gs[1, col])
    lM        = stats.norm.pdf(x_range, mmu, msd) * prior_M
    lB        = stats.norm.pdf(x_range, bmu, bsd) * prior_B
    posterior = lM / (lM + lB + 1e-12)

    ax_bot.plot(x_range, posterior, color="#222222", lw=2)
    ax_bot.axhline(0.5, color="gray", lw=1, linestyle="--", alpha=0.7,
                   label="Decision threshold (0.5)")
    ax_bot.fill_between(x_range, posterior, 0.5,
                        where=(posterior >= 0.5), alpha=0.18, color=RED,
                        label="Predict Malignant")
    ax_bot.fill_between(x_range, posterior, 0.5,
                        where=(posterior < 0.5),  alpha=0.18, color=BLUE,
                        label="Predict Benign")
    ax_bot.set_ylim(0, 1)
    ax_bot.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_bot.set_yticklabels(["0", ".25", ".50", ".75", "1"], fontsize=7)
    ax_bot.set_ylabel("P(Malignant | x)", fontsize=8)
    ax_bot.set_xlabel("Feature value", fontsize=8)
    ax_bot.tick_params(labelsize=7)
    ax_bot.set_facecolor("#EAECF0")
    ax_bot.grid(True, color="white", linewidth=0.6)
    ax_bot.set_title("Posterior — Bayes preview", fontsize=8)

handles2 = [
    plt.Rectangle((0,0),1,1, color=BLUE, alpha=0.6, label="Benign data"),
    plt.Rectangle((0,0),1,1, color=RED,  alpha=0.6, label="Malignant data"),
    plt.Line2D([0],[0], color=BLUE, lw=2, linestyle="--", label="N(Benign)"),
    plt.Line2D([0],[0], color=RED,  lw=2, linestyle="--", label="N(Malignant)"),
    plt.Line2D([0],[0], color="gray", lw=1.5, linestyle="--", label="0.5 threshold"),
]
fig.legend(handles=handles2, loc="upper right", fontsize=9,
           ncol=5, bbox_to_anchor=(1.01, .98))
plt.savefig("outputs/fig2_top5_deepdive.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("    Saved → outputs/fig2_top5_deepdive.png")

# 7. FIGURE 3 — Cohen's d separation bar chart
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor(BG)
feats_sorted = [f for f, _ in ranked]
d_vals       = [sep_scores[f] for f in feats_sorted]
colors       = [RED if d > 1.5 else ("#F5A623" if d > 0.8 else BLUE) for d in d_vals]
labels       = [f.replace("_mean","").replace("_"," ").title() for f in feats_sorted]

bars = ax.barh(labels[::-1], d_vals[::-1], color=colors[::-1],
               edgecolor="white", height=0.65)
ax.axvline(1.5, color=RED,     lw=1.5, linestyle="--", alpha=0.7, label="Large (d=1.5)")
ax.axvline(0.8, color="#F5A623", lw=1.5, linestyle="--", alpha=0.7, label="Moderate (d=0.8)")
for bar, d in zip(bars, d_vals[::-1]):
    ax.text(bar.get_width() + 0.04, bar.get_y() + bar.get_height()/2,
            f"d={d:.2f}", va="center", fontsize=9)
ax.set_xlabel("Cohen's d  (standardized group separation)", fontsize=10)
ax.set_title("Figure 3 – Feature Separation Between Benign & Malignant Groups\n"
             "Larger d → more useful for Bayesian classification",
             fontsize=11, fontweight="bold")
ax.set_facecolor("#EAECF0")
ax.grid(True, color="white", linewidth=0.7, axis="x")
ax.set_xlim(0, max(d_vals) + 0.4)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("outputs/fig3_cohens_d.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("    Saved → outputs/fig3_cohens_d.png")

# 8. WRITTEN OBSERVATIONS
print(f"""
  WRITTEN OBSERVATIONS

DATASET OVERVIEW
  • 569 biopsies: 357 benign (62.7%), 212 malignant (37.3%)
  • 30 continuous features — we focus on the 10 "mean" features
  • Zero missing values across all features
  • Empirical prior: P(Malignant) ≈ 0.373

FEATURE SEPARATION  (Cohen's d ranking)
  ★ LARGE separation  (d > 1.5) — best Bayesian predictors:
    1. concave_points_mean  d=2.33 — malignant tumors have far more
         concave contour points; the two distributions barely overlap.
    2. perimeter_mean       d=2.12 — malignant tumors are significantly
         larger in perimeter; clear right-shift in the malignant group.
    3. radius_mean          d=2.05 — closely related to perimeter/area;
         benign group tightly clustered ~12, malignant spread ~12–28.
    4. concavity_mean       d=1.87 — malignant group right-skewed with
         a long upper tail; benign cases cluster near zero.
    5. area_mean            d=1.86 — mirrors radius/perimeter findings;
         malignant mean (~978) roughly double benign mean (~463).

  ▲ MODERATE separation  (d = 0.8–1.5):
    6. compactness_mean     d=1.45 — useful signal, but more overlap.
    7. texture_mean         d=0.95 — moderate shift, wider overlap.

  ▼ SMALL separation  (d < 0.8) — weak individual predictors:
    8. smoothness_mean      d=0.80
    9. symmetry_mean        d=0.71
   10. fractal_dimension    d=0.03 — nearly identical distributions;
         negligible predictive value on its own.

""")
