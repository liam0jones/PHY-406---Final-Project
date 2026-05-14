"""
train_classifier.py
===================
Unsupervised ML phase classifier for the 2D Bose-Hubbard model.

Discovers all wavefunction CSV files in ../output/ matching the pattern
  bose_hubbard_psi_Lx{Lx}_Ly{Ly}_n{N}.csv
and runs PCA + GMM analysis for each (lattice, filling).

Method — fully unsupervised, no prior knowledge of (U/t)_c used
----------------------------------------------------------------
  PCA + GMM
       Project each wavefunction onto 50 principal components; fit a
       2-component Gaussian Mixture Model on the leading 10 PCs.
       The phase boundary is placed at P(Mott | ψ) = 0.5, with the
       Mott component identified as the cluster with higher mean U/t.

Outputs  (../output/)
---------------------
  Per-filling plots (suffix _Lx{Lx}_Ly{Ly}_n{N}):
    pca_variance_*.png        explained-variance spectrum
    pca_trajectory_*.png      PC1 & PC2 scores vs U/t
    pca_scatter_*.png         PC1 vs PC2, coloured by GMM cluster
    order_parameter_*.png     Fock-state fidelity F = max|ψ|² vs U/t
    boundary_comparison_*.png P(Mott) and Fock fidelity vs U/t

  Aggregated:
    phase_predictions.csv     U/t, Lx, Ly, N_part, phase labels, posteriors
"""

import re
import sys
import warnings
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Mott":       "#E63946",
    "Superfluid": "#457B9D",
    "boundary":   "#2d6a4f",
    "fidelity":   "#f4a261",
    "neutral":    "#6d6875",
}


def _apply_physics_style() -> None:
    plt.rcParams.update({
        "font.family":          "serif",
        "font.size":            11,
        "axes.labelsize":       12,
        "axes.titlesize":       11,
        "axes.titleweight":     "normal",
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,
        "xtick.direction":      "in",
        "ytick.direction":      "in",
        "xtick.top":            True,
        "ytick.right":          True,
        "xtick.minor.visible":  True,
        "ytick.minor.visible":  True,
        "axes.linewidth":       0.8,
        "lines.linewidth":      1.8,
        "lines.markersize":     5,
        "legend.fontsize":      9,
        "legend.framealpha":    0.9,
        "legend.edgecolor":     "0.7",
        "figure.dpi":           150,
        "savefig.dpi":          150,
        "savefig.bbox":         "tight",
        "axes.grid":            False,
    })


_apply_physics_style()



# 1. CSV discovery and loading

def discover_csv_files(output_dir: Path) -> List[Path]:
    """Return all wavefunction CSV files, sorted by (Lx, Ly, N_part)."""
    pattern = re.compile(r"bose_hubbard_psi_Lx(\d+)_Ly(\d+)_n(\d+)\.csv")
    found = sorted(
        output_dir.glob("bose_hubbard_psi_Lx*_Ly*_n*.csv"),
        key=lambda p: tuple(int(x) for x in pattern.match(p.name).groups())
        if pattern.match(p.name) else (0, 0, 0),
    )
    if found:
        return found
    legacy = output_dir / "bose_hubbard_wavefunctions.csv"
    if legacy.exists():
        return [legacy]
    return []


def parse_csv_metadata(path: Path) -> Tuple[int, int, int]:
    """Extract (Lx, Ly, N_part) from filename, or return (0, 0, 0) for legacy."""
    m = re.match(r"bose_hubbard_psi_Lx(\d+)_Ly(\d+)_n(\d+)\.csv", path.name)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return 0, 0, 0


def load_wavefunctions(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    df = pd.read_csv(csv_path)
    U_vals = df["U_over_t"].values.astype(np.float64)
    psi_cols = sorted(
        [c for c in df.columns if c.startswith("psi_")],
        key=lambda s: int(s.split("_")[1]),
    )
    Psi = df[psi_cols].values.astype(np.float64)
    D = Psi.shape[1]

    print(f"\n{'='*60}")
    print(f"  File    : {csv_path.name}")
    print(f"  Samples : {len(U_vals)}  (U/t ∈ [{U_vals.min():.2f}, {U_vals.max():.2f}])")
    print(f"  D       : {D}")
    norms = np.linalg.norm(Psi, axis=1)
    print(f"  Norms   : min={norms.min():.6f}  max={norms.max():.6f}")
    print(f"{'='*60}\n")
    return U_vals, Psi, D



# 2. Physical order parameter (validation only, not used for training)

def fock_fidelity(Psi: np.ndarray) -> np.ndarray:
    """F = max_i |ψ_i|²  →  1 in deep MI,  1/D in deep SF."""
    return np.max(Psi**2, axis=1)



# 3. GMM utilities

def fit_gmm(features: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, np.ndarray]:
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=20,
        random_state=RANDOM_STATE,
    )
    gmm.fit(features)
    return gmm, gmm.predict(features), gmm.predict_proba(features)


def mott_component(labels: np.ndarray, U_vals: np.ndarray) -> int:
    means = [
        U_vals[labels == k].mean() if np.any(labels == k) else 0.0
        for k in range(2)
    ]
    return int(np.argmax(means))


def boundary_from_posterior(U_vals: np.ndarray, p_mott: np.ndarray) -> Optional[float]:
    order = np.argsort(U_vals)
    U_s, p_s = U_vals[order], p_mott[order]
    for i in range(len(p_s) - 1):
        if (p_s[i] - 0.5) * (p_s[i + 1] - 0.5) <= 0:
            denom = p_s[i + 1] - p_s[i] + 1e-30
            t = (0.5 - p_s[i]) / denom
            return float(U_s[i] + t * (U_s[i + 1] - U_s[i]))
    return None



# 4. PCA + GMM analysis

def pca_gmm_analysis(
    U_vals: np.ndarray,
    Psi: np.ndarray,
    n_pca: int = 50,
    n_gmm_pcs: int = 10,
) -> Dict[str, Any]:
    scaler = StandardScaler(with_std=False)
    Psi_c = scaler.fit_transform(Psi)

    actual_n_pca = min(n_pca, Psi_c.shape[1], Psi_c.shape[0])
    actual_n_gmm = min(n_gmm_pcs, actual_n_pca)

    pca = PCA(n_components=actual_n_pca, random_state=RANDOM_STATE)
    scores = pca.fit_transform(Psi_c)

    gmm, labels, posteriors = fit_gmm(scores[:, :actual_n_gmm])
    mi_comp = mott_component(labels, U_vals)
    p_mott = posteriors[:, mi_comp]
    boundary = boundary_from_posterior(U_vals, p_mott)

    if boundary is not None:
        print(f"  [PCA+GMM]  Phase boundary at  U/t ≈ {boundary:.3f}")
    else:
        print("  [PCA+GMM]  No boundary found")

    return dict(
        pca=pca,
        scaler=scaler,
        scores=scores,
        labels=labels,
        phase_labels=np.where(p_mott >= 0.5, "Mott", "Superfluid"),
        posteriors=posteriors,
        p_mott=p_mott,
        mi_comp=mi_comp,
        boundary=boundary,
    )



# 5. Plotting helpers

def _vline(ax, x, label=None, color=None, lw=1.8):
    kw = dict(color=color or COLORS["boundary"], lw=lw, ls="--", zorder=5)
    if label:
        kw["label"] = label
    ax.axvline(x, **kw)


def _save(fig, name: str, tag: str) -> None:
    fname = f"{name}{tag}.png"
    fig.savefig(str(OUTPUT_DIR / fname))
    plt.close(fig)
    print(f"✓  {fname}")


def plot_pca_variance(pca_res: Dict[str, Any], tag: str, label: str):
    ev = pca_res["pca"].explained_variance_ratio_ * 100
    cum = np.cumsum(ev)
    n_show = 10

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"PCA Explained Variance — {label}", fontsize=11)

    ax1.bar(range(1, n_show + 1), ev[:n_show],
            color=COLORS["Superfluid"], edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance (%)")
    ax1.set_xlim(0, n_show + 0.5)
    ax1.set_xticks(range(1, n_show + 1))
    ax1.set_title("Individual")

    ax2.plot(range(1, len(cum) + 1), cum, "o-",
             color=COLORS["Superfluid"], ms=4, lw=1.8)
    ax2.set_xlabel("Number of PCs")
    ax2.set_ylabel("Cumulative variance (%)")
    ax2.set_title("Cumulative")
    ax2.set_xlim(0.5, len(cum) + 0.5)

    plt.tight_layout()
    _save(fig, "pca_variance", tag)


def plot_pca_trajectory(U_vals: np.ndarray, pca_res: Dict[str, Any], tag: str, label: str):
    scores = pca_res["scores"]
    boundary = pca_res["boundary"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"PCA Scores vs $U/t$ — {label}", fontsize=11)

    for ax, pc_idx, pc_label in zip(axes, [0, 1], ["PC 1", "PC 2"]):
        sc = ax.scatter(
            U_vals, scores[:, pc_idx], c=U_vals, cmap="plasma",
            s=10, alpha=0.85, rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label=r"$U/t$", pad=0.02)
        if boundary:
            _vline(ax, boundary, label=rf"$(U/t)_c \approx {boundary:.2f}$")
            ax.legend()
        ax.set_xlabel(r"$U/t$")
        ax.set_ylabel(f"{pc_label} score")
        ax.set_title(f"{pc_label} vs $U/t$")

    plt.tight_layout()
    _save(fig, "pca_trajectory", tag)


def plot_pca_scatter(U_vals: np.ndarray, pca_res: Dict[str, Any], tag: str, label: str):
    scores = pca_res["scores"]
    phase_labels = pca_res["phase_labels"]
    boundary = pca_res["boundary"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"PCA Scatter — {label}", fontsize=11)

    ax = axes[0]
    for phase in ("Superfluid", "Mott"):
        mask = phase_labels == phase
        if phase=="Mott":
            lab = "Mott Insulator"
        else:
            lab = "Superfluid"
        ax.scatter(
            scores[mask, 0], scores[mask, 1],
            c=COLORS[phase], s=14, alpha=0.75, label=lab, rasterized=True,
        )
    title = (rf"GMM clusters  $\left((U/t)_c \approx {boundary:.2f}\right)$"
             if boundary else "GMM clusters")
    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="lower right")

    ax2 = axes[1]
    sc = ax2.scatter(
        scores[:, 0], scores[:, 1], c=U_vals, cmap="plasma",
        s=14, alpha=0.85, rasterized=True,
    )
    plt.colorbar(sc, ax=ax2, label=r"$U/t$", pad=0.02)
    ax2.set_title(r"Coloured by $U/t$")
    ax2.set_xlabel("PC 1")
    ax2.set_ylabel("PC 2")

    plt.tight_layout()
    _save(fig, "pca_scatter", tag)


def plot_order_parameter(
    U_vals: np.ndarray,
    Psi: np.ndarray,
    pca_res: Dict[str, Any],
    tag: str,
    label: str,
):
    F = fock_fidelity(Psi)
    D = Psi.shape[1]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(U_vals, F, c=U_vals, cmap="plasma", s=12, alpha=0.85,
                    zorder=3, rasterized=True)
    plt.colorbar(sc, ax=ax, label=r"$U/t$", pad=0.02)
    ax.axhline(1.0, color="gray", ls=":", lw=1.0, label=r"MI limit  $F = 1$")

    if pca_res["boundary"]:
        _vline(ax, pca_res["boundary"],
               label=rf"PCA+GMM  $(U/t)_c \approx {pca_res['boundary']:.2f}$")

    ax.set_xlabel(r"$U/t$")
    ax.set_ylabel(r"Fock fidelity")
    ax.set_title(
        f"Physical Order Parameter — {label}"
    )
    ax.legend()
    ax.set_ylim(-0.02, 1.05)

    plt.tight_layout()
    _save(fig, "order_parameter", tag)


def plot_boundary_comparison(
    U_vals: np.ndarray,
    Psi: np.ndarray,
    pca_res: Dict[str, Any],
    tag: str,
    label: str,
):
    F = fock_fidelity(Psi)
    F_norm = (F - F.min()) / (F.max() - F.min() + 1e-30)
    order = np.argsort(U_vals)
    ref = pca_res["boundary"] if pca_res["boundary"] else U_vals.mean()

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(U_vals[order], pca_res["p_mott"][order],
            color=COLORS["Mott"], lw=2,
            label=r"$P(\mathrm{Mott}\,|\,\psi)$  PCA+GMM")
    ax.fill_between(U_vals[order], pca_res["p_mott"][order],
                    alpha=0.12, color=COLORS["Mott"])
    ax.plot(U_vals[order], F_norm[order],
            color=COLORS["fidelity"], lw=1.8, ls=":",
            label=r"Fock fidelity $F$")
    ax.axhline(0.5, color="gray", lw=1.0, ls="--",
               label=r"Decision  $P = 0.5$")
    ax.text((float(U_vals.min()) + ref) / 2, 0.90, "Superfluid",
            ha="center", fontsize=11,
            color=COLORS["Superfluid"], alpha=0.85, fontweight="bold")
    ax.text((ref + float(U_vals.max())) / 2, 0.90, "Mott Insulator",
            ha="center", fontsize=10,
            color=COLORS["Mott"], alpha=0.85, fontweight="bold")

    ax.set_xlabel(r"$U/t$")
    ax.set_ylabel("Probability / Normalized fidelity")
    ax.set_title(f"Phase Boundary Detection — PCA+GMM\n{label}")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(loc="center right")

    plt.tight_layout()
    _save(fig, "boundary_comparison", tag)



# 6. Save/append predictions CSV

def make_predictions_df(
    U_vals: np.ndarray,
    Lx: int, Ly: int, N_part: int,
    pca_res: Dict[str, Any],
) -> pd.DataFrame:
    return pd.DataFrame({
        "Lx": Lx,
        "Ly": Ly,
        "N_part": N_part,
        "filling": N_part / (Lx * Ly),
        "U_over_t": U_vals,
        "p_mott_pca": pca_res["p_mott"],
        "phase_pca": pca_res["phase_labels"],
    })



# 7. Per-filling analysis

def analyse_one(csv_path: Path) -> Optional[pd.DataFrame]:
    Lx, Ly, N_part = parse_csv_metadata(csv_path)
    M = Lx * Ly if Lx > 0 else 0
    if M > 0:
        label = f"{Lx}×{Ly} PBC, N={N_part}, n={N_part/M:.3g}"
        tag   = f"_Lx{Lx}_Ly{Ly}_n{N_part}"
    else:
        label = csv_path.stem
        tag   = ""

    U_vals, Psi, D = load_wavefunctions(csv_path)

    print(f"{'─'*60}")
    print(f"  PCA + GMM")
    print(f"{'─'*60}")
    pca_res = pca_gmm_analysis(U_vals, Psi)

    plot_pca_variance(pca_res, tag, label)
    plot_pca_trajectory(U_vals, pca_res, tag, label)
    plot_pca_scatter(U_vals, pca_res, tag, label)
    plot_order_parameter(U_vals, Psi, pca_res, tag, label)
    plot_boundary_comparison(U_vals, Psi, pca_res, tag, label)

    print(f"\n{'='*60}")
    print(f"  RESULT  {label}")
    print(f"{'='*60}")
    print(f"  D = {D}   U/t ∈ [{U_vals.min():.2f}, {U_vals.max():.2f}]")
    print(f"  PCA+GMM   (U/t)_c = "
          + (f"{pca_res['boundary']:.3f}" if pca_res["boundary"] else "not found"))
    print()

    return make_predictions_df(U_vals, Lx, Ly, N_part, pca_res)



# 8. Main

def main():
    csv_files = discover_csv_files(OUTPUT_DIR)
    if not csv_files:
        sys.exit(
            f"ERROR: No wavefunction CSV files found in {OUTPUT_DIR}.\n"
            "Run  julia julia/generate_data.jl  first."
        )

    print(f"\nFound {len(csv_files)} wavefunction file(s):")
    for f in csv_files:
        print(f"  {f.name}")

    all_dfs: List[pd.DataFrame] = []
    for csv_path in csv_files:
        df = analyse_one(csv_path)
        if df is not None:
            all_dfs.append(df)

    if all_dfs:
        pred_df = pd.concat(all_dfs, ignore_index=True)
        out_path = OUTPUT_DIR / "phase_predictions.csv"
        pred_df.to_csv(out_path, index=False)
        print(f"✓  phase_predictions.csv  ({len(pred_df)} rows, "
              f"{pred_df[['Lx','Ly','N_part']].drop_duplicates().shape[0]} filling(s))")


main()
