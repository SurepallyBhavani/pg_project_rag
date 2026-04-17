import json
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "comparison_results"

def get_latest_results():
    all_dirs = glob.glob(str(RESULTS_DIR / "*"))
    dir_paths = [d for d in all_dirs if os.path.isdir(d)]
    if not dir_paths:
        return None
    latest_dir = max(dir_paths, key=os.path.getmtime)
    return Path(latest_dir)

def main():
    latest_dir = get_latest_results()
    if not latest_dir:
        print("No results found.")
        return

    print(f"Generating graphs for {latest_dir}...")
    judgments_path = latest_dir / "judgments.json"
    judgments = json.loads(judgments_path.read_text(encoding="utf-8"))

    systems_data = {
        "eduassist": {"faithfulness": [], "answer_relevance": [], "context_precision": [], "overall_quality": []},
        "baseline_rag": {"faithfulness": [], "answer_relevance": [], "context_precision": [], "overall_quality": []},
        "chatgpt": {"faithfulness": [], "answer_relevance": [], "context_precision": [], "overall_quality": []},
        "notebooklm": {"faithfulness": [], "answer_relevance": [], "context_precision": [], "overall_quality": []},
    }
    wins = {"eduassist": 0, "baseline_rag": 0, "chatgpt": 0, "notebooklm": 0}

    for item in judgments:
        j = item.get("judgment", {})
        sys = j.get("systems", {})
        for s_name, metrics in sys.items():
            if not isinstance(metrics, dict): continue
            
            # Convert 0-5 scores to 0-100%
            f = (float(metrics.get("faithfulness", 0) or 0) / 5.0) * 100
            ar = (float(metrics.get("answer_relevance", 0) or 0) / 5.0) * 100
            oq = (float(metrics.get("overall_quality", 0) or 0) / 5.0) * 100
            
            systems_data[s_name]["faithfulness"].append(f)
            systems_data[s_name]["answer_relevance"].append(ar)
            systems_data[s_name]["overall_quality"].append(oq)
            
            cp = metrics.get("context_precision")
            if cp is not None:
                systems_data[s_name]["context_precision"].append((float(cp) / 5.0) * 100)
                
        best = j.get("best_overall")
        if best and best in wins:
            wins[best] += 1

    sys_names = ["EduAssist", "Baseline RAG", "ChatGPT", "NotebookLM"]
    keys = ["eduassist", "baseline_rag", "chatgpt", "notebooklm"]

    # 1. Bar Chart: Average Generation Metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sys_names))
    width = 0.25

    f_avgs = [np.mean(systems_data[k]["faithfulness"]) for k in keys]
    ar_avgs = [np.mean(systems_data[k]["answer_relevance"]) for k in keys]
    oq_avgs = [np.mean(systems_data[k]["overall_quality"]) for k in keys]

    rects1 = ax.bar(x - width, f_avgs, width, label='Faithfulness', color='#4CAF50')
    rects2 = ax.bar(x, ar_avgs, width, label='Answer Relevance', color='#2196F3')
    rects3 = ax.bar(x + width, oq_avgs, width, label='Overall Quality', color='#FF9800')

    ax.set_ylabel('Score (%)')
    ax.set_title('Average Performance Metrics (Generation)')
    ax.set_xticks(x)
    ax.set_xticklabels(sys_names)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.set_ylim(0, 110)

    # Add numeric labels
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90, fontsize=9)

    plt.tight_layout()
    plt.savefig(latest_dir / "generation_metrics.png", dpi=300)
    plt.close()

    # 2. Context Precision Chart (RAG only)
    fig, ax = plt.subplots(figsize=(6, 5))
    rag_keys = ["eduassist", "baseline_rag"]
    rag_names = ["EduAssist", "Baseline RAG"]
    
    cp_avgs = []
    for k in rag_keys:
        lst = systems_data[k]["context_precision"]
        cp_avgs.append(np.mean(lst) if lst else 0.0)

    bars = ax.bar(rag_names, cp_avgs, color=['#9C27B0', '#607D8B'], width=0.5)
    ax.set_ylabel('Context Precision (%)')
    ax.set_title('Retrieval Effectiveness (Context Precision)')
    ax.set_ylim(0, 100)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(latest_dir / "context_precision.png", dpi=300)
    plt.close()

    # 3. Pie Chart: Win Rate
    fig, ax = plt.subplots(figsize=(6, 6))
    win_counts = [wins[k] for k in keys]
    # Filter out 0 wins for cleaner pie
    labels_filtered = [name for name, count in zip(sys_names, win_counts) if count > 0]
    sizes_filtered = [count for count in win_counts if count > 0]
    colors_full = ['#E91E63', '#00BCD4', '#FFC107', '#9E9E9E']
    colors_filtered = [colors_full[i] for i, count in enumerate(win_counts) if count > 0]

    ax.pie(sizes_filtered, labels=labels_filtered, autopct='%1.1f%%', startangle=90, colors=colors_filtered)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title('Direct Query Winning Percentage')
    
    plt.tight_layout()
    plt.savefig(latest_dir / "win_rate.png", dpi=300)
    plt.close()
    
    print("Graphs generated successfully.")

if __name__ == "__main__":
    main()
