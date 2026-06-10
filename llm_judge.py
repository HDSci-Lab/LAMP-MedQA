"""
Standalone LLM-as-a-Judge Evaluation Script
Evaluates semantic similarity between predictions and references using GPT-4.
"""

import json
import os
import csv
import glob
import argparse
from datetime import datetime
from typing import List, Dict
from openai import OpenAI
import random

# Reproducibility
SEED = 42

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)


def llm_judge_score(preds: List[str], refs: List[str], client: OpenAI) -> float:
    """LLM-as-a-judge semantic similarity score (1-5 Likert scale)."""
    scores = []
    for i, (pred, ref) in enumerate(zip(preds, refs)):
        if not pred or not ref:
            scores.append(1.0)
            continue
        try:
            judge_prompt = f"""You are an expert medical QA judge. Evaluate how well the prediction matches the reference answer, using the following defined scale:

1 - Not semantically similar at all
2 - Not very semantically similar
3 - Somewhat semantically similar
4 - Quite semantically similar
5 - Very semantically similar

Reference Answer (gold standard):
{ref}

Predicted Answer (to evaluate):
{pred}

Return ONLY a single JSON object with no markdown formatting: {{"similarity": <score>}}
"""

            response = client.chat.completions.create(
                model="gpt-4",
                max_tokens=150,
                temperature=0.0,
                seed=SEED,
                messages=[{"role": "user", "content": judge_prompt}],
            )
            result = json.loads(response.choices[0].message.content.strip())
            score = result.get("similarity", 1)
            scores.append(float(max(1, min(5, score))))
        except Exception as e:
            print(f"  LLM judge error on example {i}: {e}")
            scores.append(1.0)

        if (i + 1) % 10 == 0:
            print(f"    LLM judge progress: {i+1}/{len(preds)} (avg: {sum(scores)/len(scores):.4f})")

    return sum(scores) / len(scores) if scores else 0.0


def evaluate_predictions(predictions: List[Dict], client: OpenAI) -> List[float]:
    """Evaluate a list of predictions."""
    scores = []
    total = len(predictions)

    for i, pred in enumerate(predictions):
        prediction = pred.get("prediction", "")
        reference = pred.get("reference", "")

        if not prediction or not reference:
            scores.append(1.0)
            continue

        score = llm_judge_score([prediction], [reference], client)
        scores.append(score)

        if (i + 1) % 10 == 0 or i == total - 1:
            print(f"    Progress: {i+1}/{total} (avg: {sum(scores)/len(scores):.4f})")

    return scores


def load_predictions_file(filepath: str) -> Dict:
    """Load and parse a predictions JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
        
        # Handle malformed JSON
        if content.startswith('"'):
            content = "{" + content
        if content.endswith(","):
            content = content[:-1]
        if content.endswith("]") and not content.endswith("]}"):
            content = content + "}"
        
        return json.loads(content)


def run_llm_judge(predictions_path: str, output_dir: str = "results"):
    """
    Run LLM judge evaluation on prediction files.
    
    Args:
        predictions_path: Path to JSON file or directory containing JSON files
        output_dir: Directory to save results
    """
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("OpenAI client initialized.\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find prediction files
    if os.path.isfile(predictions_path):
        prediction_files = [predictions_path]
    elif os.path.isdir(predictions_path):
        prediction_files = glob.glob(os.path.join(predictions_path, "*.json"))
    else:
        print(f"Error: {predictions_path} not found")
        return
    
    print(f"Found {len(prediction_files)} prediction file(s)\n")
    
    all_results = {}
    
    for pred_file in prediction_files:
        print(f"{'='*60}")
        print(f"Processing: {os.path.basename(pred_file)}")
        print(f"{'='*60}")
        
        try:
            data = load_predictions_file(pred_file)
        except Exception as e:
            print(f"  Error loading file: {e}")
            continue
        
        # Handle both formats
        if isinstance(data, dict):
            for model_name, predictions in data.items():
                if isinstance(predictions, dict) and "error" in predictions:
                    print(f"  Skipping {model_name}: {predictions['error']}")
                    continue
                
                if not isinstance(predictions, list):
                    continue
                
                print(f"\n  Evaluating: {model_name} ({len(predictions)} examples)")
                scores = evaluate_predictions(predictions, client)
                avg_score = sum(scores) / len(scores) if scores else 0.0
                
                all_results[model_name] = {
                    "llm_judge_score": avg_score,
                    "num_examples": len(predictions),
                    "scores": scores
                }
                
                print(f"  {model_name}: LLM Judge Score = {avg_score:.4f}")
        else:
            # Single list - use filename as model name
            model_name = os.path.splitext(os.path.basename(pred_file))[0]
            predictions = data
            
            print(f"\n  Evaluating: {model_name} ({len(predictions)} examples)")
            scores = evaluate_predictions(predictions, client)
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            all_results[model_name] = {
                "llm_judge_score": avg_score,
                "num_examples": len(predictions),
                "scores": scores
            }
            
            print(f"  {model_name}: LLM Judge Score = {avg_score:.4f}")
    
    # Save detailed results (with individual scores)
    detailed_path = os.path.join(output_dir, f"llm_judge_detailed_{timestamp}.json")
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to {detailed_path}")
    
    # Save summary CSV
    csv_path = os.path.join(output_dir, f"llm_judge_{timestamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "LLM_Judge_Score", "Num_Examples"])
        for model_name, result in all_results.items():
            writer.writerow([
                model_name,
                f"{result['llm_judge_score']:.4f}",
                result['num_examples']
            ])
    print(f"Summary CSV saved to {csv_path}")
    
    # Print final comparison
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"{'Model':<40} {'LLM Judge Score':<15} {'Examples':<10}")
    print("-" * 65)
    for model_name, result in sorted(all_results.items(), key=lambda x: x[1]['llm_judge_score'], reverse=True):
        print(f"{model_name:<40} {result['llm_judge_score']:<15.4f} {result['num_examples']:<10}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Evaluation")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON file or directory")
    parser.add_argument("--output", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    run_llm_judge(
        predictions_path=args.predictions,
        output_dir=args.output
    )
