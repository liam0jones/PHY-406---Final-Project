"""

Visualize the SF-MI phase boundary for the 2D Bose-Hubbard model.

Reads  ../output/phase_predictions.csv
  Columns: Lx | Ly | N_part | filling | U_over_t | p_mott_pca | phase_pca

For a single (Lx, Ly, filling) combination the output matches the original
single-filling plots.  When multiple fillings are present a Mott-lobe diagram
is added showing detected boundaries in the (U/t, n) plane.

Outputs
-------
  phase_diagram_{tag}.png         colour-strip phase diagram per filling
  phase_probabilities_{tag}.png   P(Mott) curve per filling
  summary_figure_{tag}.png        4-panel summary per filling
  mott_lobes.png                  2-D lobe diagram (only when ≥2 fillings)
  system_comparison_n{n}.png      finite-size scaling (only when ≥2 lattice sizes)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).parent.parent / "output"

COLOR_MOTT = "#E63946"
COLOR_SF   = "#457B9D"
COLOR_PCA  = "#457B9D"


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



def boundary_from_col(df: pd.DataFrame, col: str) -> float | None:
    s = df.sort_values("U_over_t")
    U: np.ndarray = s["U_over_t"].to_numpy(dtype=float)
    p: np.ndarray = s[col].to_numpy(dtype=float)
    for i in range(len(p) - 1):
        if (p[i] - 0.5) * (p[i + 1] - 0.5) <= 0:
            denom = p[i + 1] - p[i] + 1e-30
            t = (0.5 - p[i]) / denom
            return float(U[i] + t * (U[i + 1] - U[i]))
    return None


def _lattice_label(Lx: int, Ly: int, N_part: int) -> str:
    M = Lx * Ly
    n = N_part / M
    return f"{Lx}×{Ly} PBC, N={N_part}  (n = {n:.3g})"


def _tag(Lx: int, Ly: int, N_part: int) -> str:
    return f"_Lx{Lx}_Ly{Ly}_n{N_part}"



def plot_phase_probabilities(sub: pd.DataFrame, bdry_pca: float | None,
                              label: str, tag: str) -> None:
    s = sub.sort_values("U_over_t")
    U: np.ndarray     = s["U_over_t"].to_numpy(dtype=float)
    p_pca: np.ndarray = s["p_mott_pca"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(U, p_pca, color=COLOR_PCA, lw=2,
            label=r"$P(\mathrm{Mott}\,|\,\psi)$  PCA+GMM")
    ax.plot(U, 1 - p_pca, color=COLOR_SF, lw=2, ls="--", alpha=0.7,
            label=r"$P(\mathrm{SF}\,|\,\psi)$  PCA+GMM")
    ax.fill_between(U, p_pca, alpha=0.08, color=COLOR_MOTT)  # type: ignore[arg-type]
    ax.axhline(0.5, color="gray", lw=1.0, ls=":", label=r"$P = 0.5$")

    if bdry_pca is not None:
        ax.axvline(bdry_pca, color=COLOR_PCA, lw=1.8, ls="--", alpha=0.8,
                   label=rf"$(U/t)_c = {bdry_pca:.2f}$")
        ax.text(bdry_pca / 2, 0.93, "SF", fontsize=13,
                ha="center", color=COLOR_SF)
        ax.text((bdry_pca + float(U.max())) / 2, 0.93, "MI", fontsize=13,
                ha="center", color=COLOR_MOTT)

    ax.set_xlabel(r"$U/t$")
    ax.set_ylabel("Phase probability")
    ax.set_title(
        r"Phase Probabilities vs $U/t$ — PCA+GMM Classifier"
        f"\n{label}"
    )
    ax.set_ylim(-0.05, 1.08)
    ax.legend()

    plt.tight_layout()
    out = OUTPUT_DIR / f"phase_probabilities{tag}.png"
    fig.savefig(str(out))
    plt.close(fig)
    print(f"✓  {out.name}")



# Multi-filling Mott lobe diagram

def plot_mott_lobes(df: pd.DataFrame) -> None:
    """Plot Mott lobe boundaries in (U/t, n) plane, one curve per system size.

    Boundaries for each (Lx, Ly) lattice are shown as separate curves.  The
    largest-system-size curve is used to shade the SF region (left) and Mott
    region (right) in their respective colours.
    """
    # Collect boundary points grouped by system size (Lx, Ly).
    size_boundaries: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for key, sub in df.groupby(["Lx", "Ly", "N_part"]):
        Lx, Ly, N_part = int(key[0]), int(key[1]), int(key[2])  # type: ignore[index]
        M = Lx * Ly
        n = N_part / M
        bdry = boundary_from_col(sub, "p_mott_pca")
        if bdry is not None:
            size_boundaries.setdefault((Lx, Ly), []).append((bdry, n))

    if not size_boundaries:
        return

    # Sort sizes by total sites M (ascending); largest last.
    sorted_sizes = sorted(size_boundaries.keys(), key=lambda k: k[0] * k[1])
    largest_key = sorted_sizes[-1]

    U_max = 40.0
    all_n = [p[1] for pts in size_boundaries.values() for p in pts]
    n_lo = min(all_n) - 0.1
    n_hi = max(all_n) + 0.1

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ref_pts = sorted(size_boundaries[largest_key], key=lambda p: p[1])
    U_ref = np.array([p[0] for p in ref_pts])
    n_mid = float(np.mean(all_n))
    ax.text(float(np.min(U_ref)) * 0.45, n_mid, "Superfluid",
            ha="center", va="center", fontsize=13,
            color=COLOR_SF, alpha=0.85, fontweight="bold")
    ax.text((float(np.max(U_ref)) + U_max) / 2, n_mid, "Mott Insulator",
            ha="center", va="center", fontsize=13,
            color=COLOR_MOTT, alpha=0.85, fontweight="bold")

    # ── One curve per system size ────────────────────────────────────────────
    n_sizes = len(sorted_sizes)
    palette = plt.cm.viridis(np.linspace(0.15, 0.85, n_sizes))

    for i, size_key in enumerate(sorted_sizes):
        Lx, Ly = size_key
        M = Lx * Ly
        pts = sorted(size_boundaries[size_key], key=lambda p: p[1])
        U_vals = np.array([p[0] for p in pts])
        n_vals = np.array([p[1] for p in pts])

        is_largest = size_key == largest_key
        ax.plot(U_vals, n_vals,
                color=palette[i],
                lw=2.2 if is_largest else 1.6,
                ls="-",
                marker="o", ms=6,
                markeredgecolor="white", markeredgewidth=0.8,
                label=f"{Lx}×{Ly}  ($M={M}$)",
                zorder=5)

    ax.set_xlabel(r"Interaction strength  $U/t$")
    ax.set_ylabel(r"Filling")
    ax.set_title(
        "Mott Lobe Boundaries Detected via unsupervised ML"
    )
    ax.legend(title="System size", loc="upper left", fontsize=9)
    ax.set_xlim(0, U_max)
    ax.set_ylim(n_lo, n_hi)

    plt.tight_layout()
    out = OUTPUT_DIR / "mott_lobes.png"
    fig.savefig(str(out))
    plt.close(fig)
    print(f"✓  {out.name}")



# Cross-lattice comparison at fixed filling

def plot_system_comparison(df: pd.DataFrame, n_target: float = 1.0) -> None:
    """Two-panel figure comparing detected (U/t)_c across lattice sizes."""
    QMC_REF = 16.74
    tol = 0.05
    sub = df[(df["filling"] - n_target).abs() < tol].copy()
    if sub.empty:
        return

    systems: list[dict] = []
    for key, grp in sub.groupby(["Lx", "Ly", "N_part"]):
        Lx_i, Ly_i = int(key[0]), int(key[1])  # type: ignore[index]
        M = Lx_i * Ly_i
        bdry_pca = boundary_from_col(grp, "p_mott_pca")
        systems.append(dict(
            tick=f"{Lx_i}×{Ly_i}\n(M={M})",
            M=M, inv_M=1.0 / M,
            bdry_pca=bdry_pca,
        ))
    systems.sort(key=lambda s: s["M"])
    if not systems:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        rf"System Size Comparison at Filling $n = {n_target}$",
        fontsize=11,
    )

    inv_M = [s["inv_M"] for s in systems]
    pca_y = [s["bdry_pca"] for s in systems]

    ax1.axhline(QMC_REF, color="gray", lw=1.2, ls=":",
                label=rf"QMC (thermo. limit)  $(U/t)_c = {QMC_REF}$")
    pca_pts = [(x, y) for x, y in zip(inv_M, pca_y) if y is not None]
    if pca_pts:
        xs, ys = zip(*pca_pts)
        ax1.plot(xs, ys, "o-", color=COLOR_PCA, lw=2, ms=8, label="PCA+GMM",
                 zorder=4, markeredgecolor="white", markeredgewidth=1)

    ax1.set_xticks(inv_M)
    ax1.set_xticklabels([s["tick"] for s in systems], fontsize=9)
    ax1.set_xlabel(r"Inverse system size  $1/M$")
    ax1.set_ylabel(r"Detected  $(U/t)_c$")
    ax1.set_title("Finite-size scaling")
    ax1.legend(loc="center right")

    x_pos = np.arange(len(systems))
    pca_err = [abs(s["bdry_pca"] - QMC_REF) if s["bdry_pca"] is not None else 0.0
               for s in systems]

    bars_pca = ax2.bar(x_pos, pca_err, 0.5,
                       color=COLOR_PCA, label="PCA+GMM", edgecolor="white")
    for bar, val in zip(bars_pca, pca_err):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([s["tick"] for s in systems], fontsize=9)
    ax2.set_ylabel(
        r"$|(U/t)_{c,\mathrm{PCA}} - (U/t)_{c,\mathrm{QMC}}|$"
    )
    ax2.set_title(
        r"Deviation from QMC reference $(U/t)_c = 16.74$",
    )
    ax2.set_ylim(0, 4.2)

    plt.tight_layout()
    out = OUTPUT_DIR / f"system_comparison_n{int(round(n_target))}.png"
    fig.savefig(str(out))
    plt.close(fig)
    print(f"✓  {out.name}")




def main() -> None:
    pred_path = OUTPUT_DIR / "phase_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"{pred_path} not found.\n"
            "Run  python python/train_classifier.py  first."
        )

    df = pd.read_csv(pred_path)
    print(f"\nLoaded {len(df)} prediction rows")

    for col, default in [("Lx", 0), ("Ly", 0), ("N_part", 0)]:
        if col not in df.columns:
            df[col] = default

    groups = df.groupby(["Lx", "Ly", "N_part"])
    n_groups = len(groups)
    print(f"  {n_groups} filling group(s): "
          + ", ".join(f"N={int(k[2])}" for k in groups.groups))  # type: ignore[index]

    for key, sub in groups:
        Lx, Ly, N_part = int(key[0]), int(key[1]), int(key[2])  # type: ignore[index]
        lbl = _lattice_label(Lx, Ly, N_part) if Lx > 0 else "lattice"
        tg  = _tag(Lx, Ly, N_part)           if Lx > 0 else ""

        bdry_pca = boundary_from_col(sub, "p_mott_pca")

        print(f"\n── {lbl}")
        if bdry_pca is not None:
            print(f"   PCA+GMM   (U/t)_c = {bdry_pca:.3f}")
        else:
            print("   PCA+GMM   boundary not found")

        plot_phase_probabilities(sub, bdry_pca, lbl, tg)

    if n_groups >= 2:
        plot_mott_lobes(df)

    n_lattices = len(df[["Lx", "Ly"]].drop_duplicates())
    if n_lattices >= 2:
        seen_fillings = df["filling"].round(2).unique()
        for n_fill in sorted(seen_fillings):
            sub_fill = df[(df["filling"] - n_fill).abs() < 0.05]
            if len(sub_fill[["Lx", "Ly"]].drop_duplicates()) >= 2:
                plot_system_comparison(df, n_target=float(n_fill))


main()
