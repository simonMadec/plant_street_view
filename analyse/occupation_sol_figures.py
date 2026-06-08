"""
Figures d'occupation du sol :
- surfaces absolues (ha) : barres groupées ;
- composition et tableau : nombre de polygones par classe et par zone.

Sortie : PDF (vectoriel) sous results/occupation_sol/ (répertoire relatif à la racine du projet).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CLASSES = ["Palmier", "Forêt", "Bâti", "Eau", "Autre"]
ZONES = ["Zone 1", "Zone 2", "Zone 3", "Zone 4"]

# Deux grilles distinctes : ajuster si les surfaces (ha) diffèrent des effectifs polygones.
# Surfaces en hectares (figure barres symlog)
_ROWS_SURFACE_HA = [
    [727, 10025, 0.15, 0, 1],
    [1295, 683, 56, 248, 1],
    [1326, 4256, 0, 60, 1],
    [1027, 11990, 2, 2004, 1],
]

# Nombre de polygones (tableau + composition relative %)
_ROWS_POLYGONES = [
    [101, 452, 28, 0, 30],
    [117, 181, 3357, 11, 31],
    [74, 655, 0, 1, 30],
    [123, 554, 193, 20, 32],
]

COLORS: dict[str, str] = {
    "Palmier": "#E6B800",
    "Forêt": "#2E7D32",
    "Bâti": "#757575",
    "Eau": "#1565C0",
    "Autre": "#5D4037",
}

# Versions légèrement adoucies pour l'impression (composition)
COLORS_SOFT: dict[str, str] = {
    "Palmier": "#D4A90A",
    "Forêt": "#267330",
    "Bâti": "#8A8A8A",
    "Eau": "#1E5A9E",
    "Autre": "#6B4E45",
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "occupation_sol"

DPI = 300


def _paper_rc() -> dict:
    """Paramètres matplotlib type article (revues, deux colonnes)."""
    return {
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFBFC",
        "grid.color": "#CCCCCC",
        "grid.linewidth": 0.6,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#DDE1E6",
    }


def load_df_surface_ha() -> pd.DataFrame:
    return pd.DataFrame(_ROWS_SURFACE_HA, index=ZONES, columns=CLASSES)


def load_df_polygones() -> pd.DataFrame:
    return pd.DataFrame(_ROWS_POLYGONES, index=ZONES, columns=CLASSES)


def plot_grouped_bars_symlog(df: pd.DataFrame, out_path: Path) -> None:
    """Barres groupées par zone ; données = surface (ha) ; symlog."""
    with plt.rc_context({**_paper_rc(), "axes.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(df.index))
        n_classes = len(CLASSES)
        bar_width = 0.14
        offsets = (np.arange(n_classes) - (n_classes - 1) / 2) * bar_width

        for j, col in enumerate(CLASSES):
            heights = df[col].to_numpy(dtype=float)
            ax.bar(
                x + offsets[j],
                heights,
                bar_width,
                label=col,
                color=COLORS[col],
                edgecolor="black",
                linewidth=0.3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(list(df.index))
        ax.set_xlabel("Zone")
        ax.set_ylabel("Surface (ha)")
        ax.set_title("Surfaces absolues par classe et par zone (échelle symétrique logarithmique)")
        ax.text(
            0.5,
            -0.12,
            "Les valeurs nulles et les très petites surfaces s’affichent dans la portion linéaire "
            "près de 0 (échelle « symlog »).",
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
            style="italic",
            color="#555555",
        )
        ax.set_yscale("symlog", linthresh=1.0, linscale=1.0)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", which="both", alpha=0.35, linestyle="-")
        ax.legend(
            title="Classe d’occupation du sol",
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            frameon=True,
            title_fontsize=9,
        )
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.18)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def plot_stacked_composition_surface_pct(df: pd.DataFrame, out_path: Path) -> None:
    """Composition relative des surfaces par zone (100 %) — pourcentages à partir des ha."""
    pct = df.div(df.sum(axis=1), axis=0) * 100.0
    n_z = len(pct.index)
    y = np.arange(n_z)
    height = 0.65
    left = np.zeros(n_z)

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(10, 4.2))

        for col in CLASSES:
            w = pct[col].to_numpy(dtype=float)
            bars = ax.barh(
                y,
                w,
                height,
                left=left,
                label=col,
                color=COLORS_SOFT[col],
                edgecolor="white",
                linewidth=0.7,
                zorder=2,
            )
            # Étiquettes % sur les segments suffisamment larges (> 8 %)
            for i, (rect, val) in enumerate(zip(bars, w)):
                if val >= 8.0:
                    cx = left[i] + val / 2.0
                    ax.text(
                        cx,
                        y[i],
                        f"{val:.0f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if col not in ("Palmier",) else "#1a1a1a",
                        fontweight="medium",
                        zorder=3,
                    )
            left = left + w

        ax.set_yticks(y)
        ax.set_yticklabels(list(pct.index))
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.grid(axis="x", linestyle="--", alpha=0.45, zorder=0)
        ax.invert_yaxis()
        leg = ax.legend(
            title="Classe d’occupation du sol",
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            frameon=True,
            ncol=1,
            handletextpad=0.5,
            title_fontsize=9,
        )
        plt.tight_layout(rect=[0, 0, 0.76, 1])
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def plot_summary_table(df: pd.DataFrame, out_path: Path) -> None:
    """Effectifs en polygones avec totaux par zone et par classe."""
    row_totals = df.sum(axis=1)
    col_totals = df.sum(axis=0)
    grand = float(col_totals.sum())

    header = [""] + CLASSES + ["Total (polygones)"]
    cells: list[list[str]] = [header]

    for zone in df.index:
        row = [zone]
        for c in CLASSES:
            v = int(df.loc[zone, c])
            row.append(str(v))
        row.append(str(int(row_totals.loc[zone])))
        cells.append(row)

    total_row = ["Total (polygones)"]
    for c in CLASSES:
        total_row.append(str(int(col_totals.loc[c])))
    total_row.append(str(int(round(grand))))
    cells.append(total_row)

    with plt.rc_context({**_paper_rc(), "axes.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.axis("off")
        ax.set_title(
            "Synthèse des effectifs — nombre de polygones par zone, par classe et total",
            fontsize=11,
            pad=12,
            fontweight="semibold",
        )

        table = ax.table(
            cellText=cells,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2.0)

        for (row, col), cell in table.get_celld().items():
            if row == 0 or col == 0:
                cell.set_facecolor("#ECEFF1")
                cell.set_text_props(weight="bold")
            if row == len(cells) - 1 or col == len(header) - 1:
                if row > 0 and col > 0:
                    cell.set_facecolor("#CFD8DC")
            if row == len(cells) - 1 and col == 0:
                cell.set_facecolor("#ECEFF1")

        plt.tight_layout()
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_ha = load_df_surface_ha()
    df_poly = load_df_polygones()

    p1 = RESULTS_DIR / "barres_symlog.pdf"
    p2 = RESULTS_DIR / "composition_pct.pdf"
    p3 = RESULTS_DIR / "tableau_synthese.pdf"

    plot_grouped_bars_symlog(df_ha, p1)
    plot_stacked_composition_surface_pct(df_ha, p2)
    plot_summary_table(df_poly, p3)

    print(f"Figure 1 (surfaces ha, barres symlog) : {p1.resolve()}")
    print(f"Figure 2 (composition surface, %) : {p2.resolve()}")
    print(f"Figure 3 (tableau polygones) : {p3.resolve()}")


if __name__ == "__main__":
    main()
