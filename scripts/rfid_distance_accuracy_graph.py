from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = (ROOT.parent / "backend").resolve()
DATA_PATHS = [
    BACKEND_ROOT / "data" / "rfid_distance_accuracy.json",
    ROOT / "data" / "rfid_distance_accuracy.json",
]
OUTPUT_PATH = ROOT / "reports" / "rfid_accuracy_vs_distance.png"


def load_data():
    for DATA_PATH in DATA_PATHS:
        if not DATA_PATH.exists():
            continue

        try:
            data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {DATA_PATH}: {exc}") from exc

        if isinstance(data, dict):
            distances = data.get("distance_m", [])
            accuracy = data.get("read_accuracy_percent", [])
            if len(distances) == len(accuracy):
                valid = [
                    (float(d), float(a))
                    for d, a in zip(distances, accuracy)
                    if d is not None and a is not None
                ]
                if valid:
                    distances_valid, accuracy_valid = zip(*valid)
                    return list(distances_valid), list(accuracy_valid)

        if isinstance(data, list):
            entries = []
            for row in data:
                if not isinstance(row, dict):
                    continue
                distance = row.get("distance_m")
                accuracy = row.get("read_accuracy_percent")
                if distance is not None and accuracy is not None:
                    entries.append((float(distance), float(accuracy)))
            if entries:
                distances, accuracies = zip(*entries)
                return list(distances), list(accuracies)

    raise FileNotFoundError("No real RFID distance/accuracy data found in backend/data or project data folder.")


def draw_graph(distances, accuracies):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
    })

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor("#f2f2f2")
    ax.set_facecolor("#f7f7f7")

    ax.plot(
        distances,
        accuracies,
        color="black",
        linewidth=2,
        marker="o",
        markerfacecolor="black",
        markersize=6,
        label="RFID Read Accuracy",
    )

    ax.set_title("RFID Read Accuracy vs Distance", fontsize=22, fontweight="bold", pad=18)
    ax.set_xlabel("Distance (meters)", fontsize=16, labelpad=10)
    ax.set_ylabel("RFID Read Accuracy (%)", fontsize=16, labelpad=12)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(0, 6, 1))
    ax.set_yticks(np.arange(0, 101, 20))

    ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.25)
    ax.tick_params(axis="both", which="major", labelsize=12)

    legend_box = ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black", facecolor="#e9e9e9")
    legend_box.get_frame().set_linewidth(1.0)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")

    fig.text(0.5, 0.02, "Figure 4: Graph comparing RFID read accuracy vs. distance.", ha="center", fontsize=13, style="italic")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.03, 0.06, 0.97, 0.96))
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"[OK] RFID distance graph saved to: {OUTPUT_PATH}")


def main():
    distances, accuracies = load_data()
    draw_graph(distances, accuracies)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)
