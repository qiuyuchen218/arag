#!/usr/bin/env python3
"""
Evaluation script for ARAG predictions.

Usage:
    python scripts/eval.py \
        --predictions results/predictions.jsonl \
        --workers 10
"""

import os
import json
import re
import string
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from arag import LLMClient, Config

logger = logging.getLogger(__name__)


def normalize_answer(s):
    """Normalize answer for comparison."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class Evaluator:
    """Evaluator for ARAG predictions."""

    def __init__(self, llm_client, predictions_path):
        self.llm_client = llm_client
        self.predictions_path = predictions_path
        self.prediction_results = self.load_predictions()

    def load_predictions(self):
        """Load predictions from file."""
        if self.predictions_path.endswith('.jsonl'):
            prediction_results = []
            with open(self.predictions_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        prediction_results.append(json.loads(line))
        else:
            with open(self.predictions_path, 'r', encoding='utf-8') as f:
                prediction_results = json.load(f)
        return prediction_results

    def calculate_llm_accuracy(self, pred_answer, gold_answer):
        """Use LLM to judge if prediction is correct."""
        system_prompt = "You are an expert evaluator."
        user_prompt = f"""Please evaluate if the generated answer is correct by comparing it with the gold answer.
Generated answer: {pred_answer}
Gold answer: {gold_answer}

The generated answer should be considered correct if it:
1. Contains the key information from the gold answer
2. Is factually accurate and consistent with the gold answer
3. Does not contain any contradicting information

Respond with ONLY 'correct' or 'incorrect'.
Response:"""

        response, _ = self.llm_client.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )

        if response.strip().lower() == "correct":
            return 1.0
        else:
            return 0.0

    def calculate_contain(self, pred_answer, gold_answer):
        """Check if gold answer is contained in prediction."""
        if not pred_answer or not gold_answer:
            return 0
        s1 = normalize_answer(pred_answer)
        s2 = normalize_answer(gold_answer)
        if s2 in s1:
            return 1
        else:
            return 0

    def evaluate_single(self, idx, prediction):
        """Evaluate single prediction."""
        pred_answer = prediction.get("pred_answer", "")
        gold_answer = prediction.get("gold_answer") or prediction.get("answer", "")

        if not isinstance(pred_answer, str):
            return idx, 0.0, 0.0, "failed"

        has_answer = pred_answer and pred_answer.strip() != ""

        if has_answer:
            llm_acc = self.calculate_llm_accuracy(pred_answer, gold_answer)
            contain_acc = self.calculate_contain(pred_answer, gold_answer)
            status = "answered"
        else:
            llm_acc = 0.0
            contain_acc = 0.0
            status = "failed"

        return idx, llm_acc, contain_acc, status

    @staticmethod
    def _utc_now():
        return datetime.now(timezone.utc).isoformat()

    def _update_trace_evaluation(self, prediction, llm_acc, contain_acc, status):
        """Attach evaluation scores to a sample trace graph when available."""
        trace_path = prediction.get("trace_path")
        if not trace_path or not os.path.exists(trace_path):
            return

        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                trace = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load trace file {trace_path}: {e}")
            return

        pred_answer = prediction.get("pred_answer", "")
        gold_answer = prediction.get("gold_answer") or prediction.get("answer", "")
        eval_metadata = {
            "pred_answer": pred_answer,
            "gold_answer": gold_answer,
            "score": llm_acc,
            "llm_accuracy": llm_acc,
            "contain_accuracy": contain_acc,
            "correct": bool(llm_acc),
            "status": status,
        }

        evaluation_node = None
        for node in reversed(trace.get("nodes", [])):
            if node.get("type") == "evaluation":
                evaluation_node = node
                break

        if evaluation_node is None:
            node_numbers = []
            for node in trace.get("nodes", []):
                node_id = str(node.get("id", ""))
                if node_id.startswith("n_"):
                    try:
                        node_numbers.append(int(node_id.split("_", 1)[1]))
                    except ValueError:
                        pass
            next_node_id = f"n_{(max(node_numbers) if node_numbers else 0) + 1:03d}"
            next_step = max([n.get("step_index", 0) for n in trace.get("nodes", [])] or [0]) + 1
            evaluation_node = {
                "id": next_node_id,
                "type": "evaluation",
                "content": None,
                "metadata": {},
                "timestamp": self._utc_now(),
                "step_index": next_step,
                "status": "success",
            }
            trace.setdefault("nodes", []).append(evaluation_node)

            answer_node = None
            for node in reversed(trace.get("nodes", [])):
                if node.get("type") == "answer":
                    answer_node = node.get("id")
                    break
            if answer_node:
                trace.setdefault("edges", []).append({
                    "source": answer_node,
                    "target": next_node_id,
                    "type": "evaluates",
                    "metadata": {},
                })

        evaluation_node.setdefault("metadata", {}).update(eval_metadata)
        evaluation_node["timestamp"] = self._utc_now()
        trace.setdefault("metadata", {}).update(eval_metadata)

        try:
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Could not update trace file {trace_path}: {e}")

    def evaluate(self, max_workers, output_dir=None):
        """Run evaluation with concurrent processing."""
        llm_scores = [0.0] * len(self.prediction_results)
        contain_scores = [0.0] * len(self.prediction_results)
        statuses = [""] * len(self.prediction_results)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.evaluate_single, idx, pred): idx
                for idx, pred in enumerate(self.prediction_results)
            }

            completed = 0
            total_llm_score = 0.0
            total_contain_score = 0.0
            answered_count = 0
            pbar = tqdm(total=len(futures), desc="Evaluating", unit="sample")

            for future in as_completed(futures):
                idx, llm_acc, contain_acc, status = future.result()
                llm_scores[idx] = llm_acc
                contain_scores[idx] = contain_acc
                statuses[idx] = status
                self.prediction_results[idx]["llm_accuracy"] = llm_acc
                self.prediction_results[idx]["contain_accuracy"] = contain_acc
                self.prediction_results[idx]["status"] = status
                self._update_trace_evaluation(
                    self.prediction_results[idx],
                    llm_acc,
                    contain_acc,
                    status,
                )

                if status == "answered":
                    answered_count += 1
                total_llm_score += llm_acc
                total_contain_score += contain_acc

                completed += 1

                if answered_count > 0:
                    current_llm_acc = total_llm_score / answered_count
                    current_contain_acc = total_contain_score / answered_count
                else:
                    current_llm_acc = 0.0
                    current_contain_acc = 0.0

                answer_rate = answered_count / completed
                pbar.set_postfix({
                    'Answered': f'{answer_rate:.1%}',
                    'LLM_Acc': f'{current_llm_acc:.3f}',
                    'Contain': f'{current_contain_acc:.3f}'
                })
                pbar.update(1)
            pbar.close()

        total_samples = len(self.prediction_results)
        answered_samples = sum(1 for s in statuses if s == "answered")
        failed_samples = sum(1 for s in statuses if s == "failed")
        answer_rate = answered_samples / total_samples if total_samples > 0 else 0

        if answered_samples > 0:
            llm_accuracy = sum(llm_scores) / answered_samples
            contain_accuracy = sum(contain_scores) / answered_samples
        else:
            llm_accuracy = 0.0
            contain_accuracy = 0.0

        total_cost = sum(p.get('total_cost', 0) for p in self.prediction_results)
        total_retrieved_tokens = sum(p.get('total_retrieved_tokens', 0) for p in self.prediction_results)
        avg_cost = total_cost / len(self.prediction_results) if self.prediction_results else 0
        avg_retrieved_tokens = total_retrieved_tokens / len(self.prediction_results) if self.prediction_results else 0
        avg_loops = sum(p.get('loops', 0) for p in self.prediction_results) / len(self.prediction_results) if self.prediction_results else 0

        logger.info(f"Evaluation Results:")
        logger.info(f"  Total Samples: {total_samples}")
        logger.info(f"  Answered: {answered_samples} ({answer_rate:.2%})")
        logger.info(f"  Failed: {failed_samples}")
        logger.info(f"  LLM Accuracy: {llm_accuracy:.4f}")
        logger.info(f"  Contain Accuracy: {contain_accuracy:.4f}")
        logger.info(f"  Total Cost: ${total_cost:.4f}")
        logger.info(f"  Avg Loops: {avg_loops:.2f}")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            actual_output_dir = output_dir
        else:
            actual_output_dir = os.path.dirname(self.predictions_path)

        base_name = os.path.basename(self.predictions_path)

        output_predictions_path = os.path.join(actual_output_dir, base_name)
        if self.predictions_path.endswith('.jsonl'):
            with open(output_predictions_path, "w", encoding="utf-8") as f:
                for pred in self.prediction_results:
                    f.write(json.dumps(pred, ensure_ascii=False) + '\n')
        else:
            with open(output_predictions_path, "w", encoding="utf-8") as f:
                json.dump(self.prediction_results, f, ensure_ascii=False, indent=4)

        if base_name.endswith('.jsonl'):
            summary_name = base_name.replace('.jsonl', '_eval_summary.json')
        elif base_name.endswith('.json'):
            summary_name = base_name.replace('.json', '_eval_summary.json')
        else:
            summary_name = "eval_summary.json"

        summary_path = os.path.join(actual_output_dir, summary_name)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_samples": total_samples,
                "answered_samples": answered_samples,
                "failed_samples": failed_samples,
                "answer_rate": round(answer_rate, 4),
                "llm_accuracy": llm_accuracy,
                "contain_accuracy": contain_accuracy,
                "correct_by_llm": int(sum(llm_scores)),
                "correct_by_contain": int(sum(contain_scores)),
                "total_cost": round(total_cost, 6),
                "avg_cost_per_question": round(avg_cost, 6),
                "total_retrieved_tokens": int(total_retrieved_tokens),
                "avg_retrieved_tokens": round(avg_retrieved_tokens, 1),
                "avg_loops": round(avg_loops, 2)
            }, f, ensure_ascii=False, indent=4)

        logger.info(f"Summary saved to: {summary_path}")

        return llm_accuracy, contain_accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARAG predictions")
    parser.add_argument("--predictions", "-p", required=True, help="Predictions file path (.json or .jsonl)")
    parser.add_argument("--workers", "-w", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print(f"\n{'='*60}")
    print(f"ARAG Evaluation")
    print(f"{'='*60}")
    print(f"Predictions: {args.predictions}")
    print(f"Workers: {args.workers}")
    print(f"{'='*60}\n")

    llm_client = LLMClient(
        model=os.getenv('ARAG_MODEL', 'gpt-4o-mini'),
        api_key=os.getenv('ARAG_API_KEY'),
        base_url=os.getenv('ARAG_BASE_URL', 'https://api.openai.com/v1')
    )

    evaluator = Evaluator(llm_client, args.predictions)
    llm_acc, contain_acc = evaluator.evaluate(max_workers=args.workers, output_dir=args.output)

    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"LLM Accuracy: {llm_acc:.4f}")
    print(f"Contain Accuracy: {contain_acc:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
