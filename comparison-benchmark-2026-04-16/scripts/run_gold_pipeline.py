import json
import os
import sys
import time
import subprocess
import csv
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()
import openai # type: ignore

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
EDUASSIST_DIR = PROJECT_ROOT / "rag-project-2026-04-09"
BASELINE_DIR = PROJECT_ROOT / "rag-baseline-2026-04-16"
DATASET_FILE = ROOT / "gold_dataset_50.json"
RESULTS_DIR = ROOT / "gold_pipeline_three_way"

def run_project_retriever(project_dir: Path, query: str) -> Dict[str, Any]:
    script = (
        "import sys\n"
        "sys.stdout.reconfigure(encoding='utf-8')\n"
        "import json\n"
        "from app import process_query\n"
        f"result = process_query({json.dumps(query)})\n"
        "answer = str(result.get('answer', ''))\n"
        "evidence = result.get('evidence', [])\n"
        "sources = [e.get('snippet', '') for e in evidence if 'snippet' in e] if isinstance(evidence, list) else []\n"
        "print(json.dumps({'answer': answer, 'context': sources}, ensure_ascii=False))\n"
    )
    completed = subprocess.run(
        ["python", "-c", script],
        cwd=str(project_dir),
        capture_output=True, text=True, encoding="utf-8", errors="ignore"
    )
    if completed.returncode != 0:
        return {"answer": "Execution failed", "context": []}
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return {"answer": "", "context": []}
    try:
        return json.loads(lines[-1])
    except:
        return {"answer": "", "context": []}

def run_chatgpt(client: openai.OpenAI, query: str) -> Dict[str, Any]:
    prompt = "You are a helpful academic assistant answering computer science questions."
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            temperature=0,
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": query}]
        )
        ans = completion.choices[0].message.content or ""
        return {"answer": ans, "context": []}
    except Exception as e:
        print(f"ChatGPT execution failed: {e}")
        return {"answer": "", "context": []}

def calc_extractive_match(client: openai.OpenAI, answer: str, must_contain: List[str], must_not: List[str]) -> float:
    if not must_contain: return 1.0
    
    prompt = (
        "You are an impartial evaluator checking if a generated academic answer contains specific factual concepts. "
        "I will give you a list of 'must_contain' facts and a 'system_answer'. "
        "Count how many of the 'must_contain' facts are successfully addressed or present in the answer CONCEPTUALLY. It does not need to be an exact string match, but the meaning must be present. "
        "Also count how many of the 'must_not_contain' concepts are present. "
        f"The total possible facts is {len(must_contain)}. "
        "Return strict JSON: {\"facts_present_count\": int, \"forbidden_words_count\": int}"
    )
    payload = {
        "must_contain_list": must_contain,
        "must_not_contain_list": must_not,
        "system_answer": answer
    }
    for attempt in [2, 4]:
        try:
            completion = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                temperature=0,
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                response_format={"type": "json_object"}
            )
            content = completion.choices[0].message.content or "{}"
            result = json.loads(content)
            matches = int(result.get("facts_present_count", 0))
            penalties = int(result.get("forbidden_words_count", 0))
            score = (min(matches, len(must_contain)) / len(must_contain)) - (penalties * 0.2)
            return max(0.0, min(1.0, score))
        except Exception:
            time.sleep(attempt)
    return 0.0

def calc_recall_k(context: List[str], targets: List[str]) -> float:
    if not targets: return 1.0
    if not context: return 0.0
    full_context = " ".join(context).lower()
    matches = sum(1 for t in targets if t.lower() in full_context)
    return matches / len(targets)

def evaluate_ragas(client: openai.OpenAI, query: str, answer: str, context: List[str]) -> Dict[str, int]:
    prompt = (
        "You are an impartial RAG evaluator. Evaluate the provided system answer against its retrieved context. "
        "Score 'faithfulness' (0-5) based on how strictly the answer adheres to the context (no hallucinations). "
        "Score 'answer_relevance' (0-5) based on how directly the answer addresses the user query. "
        "Return strict JSON only: {\"faithfulness\": int, \"answer_relevance\": int}. "
        "If there is no context given, assume faithfulness is 5."
    )
    payload = {"query": query, "system_answer": answer, "retrieved_context": "\n\n".join(context)}
    
    for attempt in [2, 4]:
        try:
            completion = client.chat.completions.create(
                model="openai/gpt-4o-mini", # Evaluator Model
                temperature=0,
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(payload)}],
                response_format={"type": "json_object"}
            )
            content = completion.choices[0].message.content or "{}"
            result = json.loads(content)
            return {"faithfulness": int(result.get("faithfulness", 0)), "answer_relevance": int(result.get("answer_relevance", 0))}
        except Exception as e:
            time.sleep(attempt)
    return {"faithfulness": 0, "answer_relevance": 0}

def build_graphs(summary_data: Dict[str, Any], out_dir: Path):
    metrics = ["extractive_avg", "recall_avg", "ragas_f_avg", "ragas_ar_avg", "total_weighted_score"]
    
    systems = list(summary_data.keys())
    # 1. Total Unified Score Graph
    fig, ax = plt.subplots(figsize=(8, 6))
    scores = [(summary_data[s]["total_weighted_score"] / summary_data[s]["queries"]) * 100 for s in systems]
    bars = ax.bar(["EduAssist", "Baseline RAG", "ChatGPT"], scores, color=['#4CAF50', '#607D8B', '#2196F3'])
    ax.set_ylabel('Unified Score (%)')
    ax.set_title('Overall Unified Pipeline Performance')
    ax.set_ylim(0, 110)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.1f}%', ha='center')
    plt.tight_layout()
    plt.savefig(out_dir / "unified_score.png", dpi=300)
    plt.close()

    # 2. Recall Graph (Exclude ChatGPT as it has no context)
    fig, ax = plt.subplots(figsize=(6, 5))
    rag_sys = ["eduassist", "baseline_rag"]
    recall = [(summary_data[s]["recall_avg"] / summary_data[s]["queries"]) * 100 for s in rag_sys]
    bars = ax.bar(["EduAssist", "Baseline RAG"], recall, color=['#9C27B0', '#607D8B'])
    ax.set_ylabel('Recall@K (%)')
    ax.set_title('Target Citation Retrieval Effectiveness')
    ax.set_ylim(0, 110)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.1f}%', ha='center')
    plt.tight_layout()
    plt.savefig(out_dir / "recall_comparison.png", dpi=300)
    plt.close()

    # 3. Grouped Multi-metric Graph
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(3)
    width = 0.25
    
    ex_match = [(summary_data[s]["extractive_avg"] / summary_data[s]["queries"]) * 100 for s in systems]
    ragas_f = [((summary_data[s]["ragas_f_avg"] / summary_data[s]["queries"]) / 5.0) * 100 for s in systems]
    ragas_ar = [((summary_data[s]["ragas_ar_avg"] / summary_data[s]["queries"]) / 5.0) * 100 for s in systems]

    b1 = ax.bar(x - width, ex_match, width, label='Extractive Match', color='#FF9800')
    b2 = ax.bar(x, ragas_f, width, label='Faithfulness', color='#8BC34A')
    b3 = ax.bar(x + width, ragas_ar, width, label='Answer Relevance', color='#03A9F4')

    ax.set_ylabel('Score (%)')
    ax.set_title('Detailed Metric Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(["EduAssist", "Baseline RAG", "ChatGPT"])
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(out_dir / "detailed_metrics.png", dpi=300)
    plt.close()

def main():
    if not DATASET_FILE.exists():
        print(f"Error: {DATASET_FILE.name} not found.")
        return 1

    dataset = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    valid_pairs = [d for d in dataset if d.get("validation_criteria", {}).get("expected_response") != "Error generating ground truth"]
    
    # Process 15 valid queries to keep run time manageable but comprehensive
    eval_pairs = valid_pairs[:15]
    
    if not eval_pairs:
        print("No valid queries found.")
        return 1
        
    api_key = os.getenv("OPENROUTER_API_KEY")
    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    
    out_dir = RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    systems = ["eduassist", "baseline_rag", "chatgpt"]
    summary_data = {
        s: {"queries": 0, "total_weighted_score": 0.0, "extractive_avg": 0.0, "recall_avg": 0.0, "ragas_f_avg": 0.0, "ragas_ar_avg": 0.0}
        for s in systems
    }
    
    all_results = []
    print(f"\n--- Running 3-Way Pipeline on {len(eval_pairs)} queries ---\n")
    
    for i, item in enumerate(eval_pairs):
        query = item["query"]
        vc = item["validation_criteria"]
        print(f"[{i+1}/{len(eval_pairs)}] Testing: {query}")
        
        row_eval = {"id": item["id"], "query": query, "systems": {}}
        
        # System 1: EduAssist
        s1 = run_project_retriever(EDUASSIST_DIR, query)
        
        # System 2: Baseline RAG
        s2 = run_project_retriever(BASELINE_DIR, query)
        
        # System 3: ChatGPT via API
        s3 = run_chatgpt(client, query)
        
        system_outputs = {
            "eduassist": s1,
            "baseline_rag": s2,
            "chatgpt": s3
        }

        for sys_name, res in system_outputs.items():
            ans = res["answer"]
            ctx = res["context"]
            
            ex = calc_extractive_match(client, ans, vc.get("must_contain", []), vc.get("must_not_contain", []))
            
            rec = 0.0
            if sys_name in ["eduassist", "baseline_rag"]:
                rec = calc_recall_k(ctx, vc.get("target_citations", []))
                
            ragas = evaluate_ragas(client, query, ans, ctx)
            f = ragas["faithfulness"]
            ar = ragas["answer_relevance"]
            
            # Weighted Scoring Rules
            # For ChatGPT: 50% Extractive, 50% RAGAS (It has no context array)
            # For RAGs: 40% Extractive, 30% Recall, 30% RAGAS
            ragas_norm = ((f + ar) / 10.0)
            if sys_name == "chatgpt":
                unified = (ex * 0.5) + (ragas_norm * 0.5)
            else:
                unified = (ex * 0.4) + (rec * 0.3) + (ragas_norm * 0.3)
                
            row_eval["systems"][sys_name] = {
                "answer_preview": ans[:100] + "...",
                "unified_score": round(unified, 2),
                "extractive_match": round(ex, 2),
                "recall": round(rec, 2) if sys_name != "chatgpt" else "N/A",
                "faithfulness": f,
                "relevance": ar
            }
            
            sd = summary_data[sys_name]
            sd["queries"] += 1
            sd["total_weighted_score"] += unified
            sd["extractive_avg"] += ex
            sd["recall_avg"] += rec
            sd["ragas_f_avg"] += f
            sd["ragas_ar_avg"] += ar

        all_results.append(row_eval)

    # Output CSV and JSON
    with open(out_dir / "evaluations.json", "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2, ensure_ascii=False)
        
    with open(out_dir / "unified_summary.csv", "w", encoding="utf-8", newline="") as cfile:
        writer = csv.writer(cfile)
        writer.writerow(["System", "Unified Score", "Extractive Match", "Recall", "Faithfulness", "Relevance"])
        for s in systems:
            q = summary_data[s]["queries"]
            writer.writerow([
                s,
                f"{(summary_data[s]['total_weighted_score']/q)*100:.1f}%",
                f"{(summary_data[s]['extractive_avg']/q)*100:.1f}%",
                f"{(summary_data[s]['recall_avg']/q)*100:.1f}%" if s != "chatgpt" else "N/A",
                f"{summary_data[s]['ragas_f_avg']/q:.2f}",
                f"{summary_data[s]['ragas_ar_avg']/q:.2f}"
            ])
            
    build_graphs(summary_data, out_dir)
    print(f"\n3-Way Pipeline Complete! Charts and unified CSV written to {out_dir}")

if __name__ == "__main__":
    raise SystemExit(main())
