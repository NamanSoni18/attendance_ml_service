import json
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta

# --- PATH CONFIGURATION ---
ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = (ROOT.parent / "backend").resolve()

# List of potential paths where your JSON logs might be saved
LOG_PATHS = [
    ROOT / "scan_results.json",
    BACKEND_ROOT / "data" / "scan_results.json",
    ROOT / "data" / "scan_history.json",
]
OUTPUT_PATH = ROOT / "reports" / "face_scan_dashboard.png"
THRESHOLD = 0.68

def read_json(path: Path):
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []

def normalize_rows(raw_rows):
    rows = []
    # Create an IST timezone object (UTC + 5 hours and 30 minutes)
    ist_tz = timezone(timedelta(hours=5, minutes=30))

    for index, row in enumerate(raw_rows):
        if not isinstance(row, dict):
            continue

        value = row.get("confidence") or row.get("score")
        if value is None:
            continue

        try:
            score = float(value)
        except (TypeError, ValueError):
            continue

        result = str(row.get("result") or row.get("status") or "unknown").lower()
        name = row.get("name") or row.get("uid") or f"Scan {index + 1}"
        
        # Extract and convert the time to IST
        scanned_at = row.get("scanned_at", "")
        if scanned_at:
            try:
                # Replace 'Z' with '+00:00' so Python parses it correctly as UTC
                clean_time_str = scanned_at.replace('Z', '+00:00')
                dt_utc = datetime.fromisoformat(clean_time_str)
                
                # Convert the UTC time to IST
                dt_ist = dt_utc.astimezone(ist_tz)
                
                # Format it beautifully as HH:MM:SS
                time_str = dt_ist.strftime('%H:%M:%S')
                label = f"{name}\n({time_str})"
            except ValueError:
                # Fallback if the timestamp string is somehow corrupted
                time_str = scanned_at.split('T')[-1][:8] if "T" in scanned_at else ""
                label = f"{name}\n({time_str})"
        else:
            label = name

        rows.append({"label": label, "score": score, "result": result})
    return rows

def load_rows():
    for path in LOG_PATHS:
        rows = normalize_rows(read_json(path))
        if rows:
            return rows
    raise ValueError(
        "No valid face-scan data found. Save real JSON scan results into one of: "
        + ", ".join(str(p) for p in LOG_PATHS)
    )

def draw_chart(rows):
    if not rows:
        raise ValueError("No scan rows available to plot.")

    # Extract data for plotting
    labels = [r["label"] for r in rows]
    scores = [r["score"] for r in rows]
    
    # Assign colors: Green for accepted (>= THRESHOLD), Red for rejected
    colors = ['#22c55e' if s >= THRESHOLD else '#ef4444' for s in scores]

    # Dynamically scale the width of the image based on how many scans exist
    # This prevents the graph from getting squeezed when data increases
    fig_width = max(10, len(rows) * 0.8)
    plt.figure(figsize=(fig_width, 7), facecolor='#f8fafc')

    # Draw the bars
    bars = plt.bar(labels, scores, color=colors, edgecolor='black', linewidth=1, zorder=3)

    # Draw the Threshold Line
    plt.axhline(y=THRESHOLD, color='#334155', linestyle='--', linewidth=2, 
                label=f'Security Threshold ({THRESHOLD})', zorder=4)

    # Add numeric score labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='#0f172a')

    # Format Titles and Labels
    plt.title('Face Recognition Confidence Tracking', fontsize=18, fontweight='bold', pad=20, color='#0f172a')
    plt.xlabel('Scan Target & Time', fontsize=13, labelpad=15, fontweight='bold')
    plt.ylabel('Cosine Similarity Score', fontsize=13, labelpad=15, fontweight='bold')
    
    # Format Axes and Grid
    plt.ylim(0, 1.1) 
    plt.xticks(rotation=45, ha='right', fontsize=10, color='#334155')
    plt.yticks(fontsize=10, color='#334155')
    plt.grid(axis='y', linestyle='-', alpha=0.3, zorder=0)
    
    # Add a legend
    plt.legend(fontsize=12, loc='upper right', framealpha=1)
    plt.tight_layout()

    # Save to disk
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] High-Resolution Dashboard saved to: {OUTPUT_PATH}")

def main():
    rows = load_rows()
    draw_chart(rows)

if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1)