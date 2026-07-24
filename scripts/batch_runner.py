#!/usr/bin/env python3
"""
Batch Runner for ARAG - Supports concurrent execution and checkpoint resume.

Usage:
    python scripts/batch_runner.py \
        --config configs/example.yaml \
        --questions data/questions.json \
        --output results/
"""

import os
import json
import argparse
import logging
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from tqdm import tqdm
from arag import LLMClient, BaseAgent, ToolRegistry, Config
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.semantic_search import SemanticSearchTool
from arag.tools.read_chunk import ReadChunkTool
from arag.utils.trace_graph import TraceGraph

logging.basicConfig(level=logging.ERROR)


class BatchRunner:
    """Batch runner with concurrent execution and checkpoint resume."""

    def __init__(
        self,
        config: Config,
        questions_file: str,
        output_dir: str,
        limit: int = None,
        num_workers: int = 10,
        verbose: bool = False
    ):
        self.config = config
        self.questions_file = Path(questions_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.num_workers = num_workers
        self.verbose = verbose

        self.predictions_file = self.output_dir / "predictions.jsonl"
        self.dataset_name = self.questions_file.stem
        self.trace_dir = self.output_dir / "traces" / self.dataset_name
        self.trace_html_dir = self.output_dir / "trace_html" / self.dataset_name
        self.write_lock = Lock()

        self.questions = self._load_questions()
        self._shared_tools = self._init_shared_tools()

        prompt_file = Path(__file__).parent.parent / "src/arag/agent/prompts/default.txt"
        if prompt_file.exists():
            self._system_prompt = prompt_file.read_text()
        else:
            self._system_prompt = "You are a helpful assistant."

    def _init_shared_tools(self) -> ToolRegistry:
        """Initialize shared tools (embedding model loaded only once)."""
        data_config = self.config.get('data', {})
        chunks_file = data_config.get('chunks_file', 'data/chunks.json')
        index_dir = data_config.get('index_dir', 'data/index')

        tools = ToolRegistry()
        tools.register(KeywordSearchTool(chunks_file=chunks_file))
        tools.register(ReadChunkTool(chunks_file=chunks_file))

        index_file = Path(index_dir) / "sentence_index.pkl"
        if index_file.exists():
            embedding_config = self.config.get('embedding', {})
            print(f"Loading embedding model: {embedding_config.get('model', 'sentence-transformers/all-MiniLM-L6-v2')}")
            tools.register(SemanticSearchTool(
                chunks_file=chunks_file,
                index_dir=index_dir,
                model_name=embedding_config.get('model', 'sentence-transformers/all-MiniLM-L6-v2'),
                device=embedding_config.get('device')
            ))
            print("Embedding model loaded successfully!")
        else:
            print(f"Warning: Index not found at {index_file}, semantic search disabled")

        return tools

    def _load_questions(self) -> List[Dict[str, Any]]:
        """Load questions from file."""
        with open(self.questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)

        if self.limit:
            questions = questions[:self.limit]

        return questions

    def _load_completed_qids(self) -> set:
        """Load completed question IDs for checkpoint resume."""
        completed_qids = set()

        if not self.predictions_file.exists():
            return completed_qids

        try:
            with open(self.predictions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if 'question' in data and 'pred_answer' in data:
                            qid = data.get('qid') or data.get('id')
                            if qid is not None:
                                completed_qids.add(qid)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Error loading completed data: {e}")

        return completed_qids

    def _append_prediction(self, prediction: Dict[str, Any]):
        """Append prediction to file (thread-safe)."""
        with self.write_lock:
            with open(self.predictions_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(prediction, ensure_ascii=False) + '\n')

    def _create_agent(self) -> BaseAgent:
        """Create agent instance with shared tools."""
        llm_config = self.config.get('llm', {})

        client = LLMClient(
            model=llm_config.get('model') or os.getenv('ARAG_MODEL', 'gpt-4o-mini'),
            api_key=llm_config.get('api_key') or os.getenv('ARAG_API_KEY'),
            base_url=llm_config.get('base_url') or os.getenv('ARAG_BASE_URL', 'https://api.openai.com/v1'),
            reasoning_effort=llm_config.get('reasoning_effort')
        )

        agent_config = self.config.get('agent', {})

        return BaseAgent(
            llm_client=client,
            tools=self._shared_tools,
            system_prompt=self._system_prompt,
            max_loops=agent_config.get('max_loops', 10),
            max_token_budget=agent_config.get('max_token_budget', 128000),
            verbose=self.verbose
        )

    @staticmethod
    def _safe_path_part(value: Any) -> str:
        text = str(value) if value is not None else "unknown"
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)
        return safe[:120] or "unknown"

    def _process_one(
        self,
        item: Dict[str, Any],
        agent: BaseAgent,
        sample_index: int = 0,
    ) -> Dict[str, Any]:
        """Process one question."""
        qid = item.get('qid') or item.get('id')
        sample_id = qid if qid is not None else f"sample_{sample_index:06d}"
        safe_sample_id = self._safe_path_part(sample_id)

        question = item.get('question', '')
        gold_answer = item.get('answer', item.get('gold_answer', ''))

        trace_logger = TraceGraph(
            sample_id=str(sample_id),
            dataset=self.dataset_name,
            metadata={
                "qid": qid,
                "sample_index": sample_index,
                "questions_file": str(self.questions_file),
            },
        )

        question_node = trace_logger.add_question(question, {
            "qid": qid,
            "sample_index": sample_index,
            "dataset": self.dataset_name,
        })

        trace_path = self.trace_dir / f"{safe_sample_id}.json"
        trace_html_path = self.trace_html_dir / f"{safe_sample_id}.html"

        try:
            result = agent.run(question, trace_logger=trace_logger)
            trace_logger.metadata.update({
                "final_answer": result.get("answer", ""),
                "pred_answer": result.get("answer", ""),
                "termination_reason": result.get("termination_reason", ""),
                "total_cost": result.get("total_cost", 0),
                "loops": result.get("loops", 0),
                "total_retrieved_tokens": result.get("total_retrieved_tokens", 0),
                "raw_error": result.get("raw_error"),
            })
            trace_logger.save_json(trace_path)
            trace_logger.save_html(trace_html_path)

            return {
                'qid': qid,
                'question': question,
                'trajectory': result['trajectory'],
                'gold_answer': gold_answer,
                'pred_answer': result['answer'],
                'total_cost': result['total_cost'],
                'loops': result['loops'],
                'total_retrieved_tokens': result.get('total_retrieved_tokens', 0),
                'retrieval_logs': result.get('retrieval_logs', []),
                'chunks_read_count': result.get('chunks_read_count', 0),
                'chunks_read_ids': result.get('chunks_read_ids', []),
                'read_chunks': result.get('read_chunks', {}),
                'search_history': result.get('search_history', []),
                'message_trace': result.get('message_trace', []),
                'final_messages': result.get('final_messages', []),
                'termination_reason': result.get('termination_reason', ''),
                'trace_path': str(trace_path),
                'trace_html_path': str(trace_html_path),
            }
        except Exception as e:
            error_answer = f"Error: {str(e)}"
            parent = trace_logger.latest_node_id("llm_call") or question_node
            trace_logger.add_error(parent, e, "batch_runner", 0, "error")
            trace_logger.add_answer(None, "", 0, "error", failed=True, raw_error=str(e))
            trace_logger.metadata.update({
                "final_answer": "",
                "pred_answer": error_answer,
                "termination_reason": "error",
                "error": str(e),
            })
            trace_logger.save_json(trace_path)
            trace_logger.save_html(trace_html_path)

            return {
                'qid': qid,
                'question': question,
                'trajectory': [],
                'gold_answer': gold_answer,
                'pred_answer': error_answer,
                'total_cost': 0,
                'loops': 0,
                'total_retrieved_tokens': 0,
                'retrieval_logs': [],
                'chunks_read_count': 0,
                'chunks_read_ids': [],
                'read_chunks': {},
                'search_history': [],
                'message_trace': [],
                'final_messages': [],
                'termination_reason': 'error',
                'error': str(e),
                'trace_path': str(trace_path),
                'trace_html_path': str(trace_html_path),
            }

    def run(self):
        """Run batch processing."""
        completed_qids = self._load_completed_qids()

        pending = [q for q in self.questions
                   if (q.get('qid') or q.get('id')) not in completed_qids]

        print(f"Total questions: {len(self.questions)}")
        print(f"Completed: {len(completed_qids)}")
        print(f"Pending: {len(pending)}")

        if not pending:
            print("All questions completed!")
            return

        print(f"Starting with {self.num_workers} workers...")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}

            for sample_index, item in enumerate(pending):
                agent = self._create_agent()
                future = executor.submit(self._process_one, item, agent, sample_index)
                futures[future] = item.get('qid') or item.get('id')

            with tqdm(total=len(pending), desc="Processing") as pbar:
                for future in as_completed(futures):
                    qid = futures[future]
                    try:
                        result = future.result()
                        self._append_prediction(result)
                    except Exception as e:
                        print(f"Error processing {qid}: {e}")
                    pbar.update(1)

        print(f"\nResults saved to: {self.predictions_file}")
        print(f"Trace JSON saved under: {self.trace_dir}")
        print(f"Trace HTML saved under: {self.trace_html_dir}")


def main():
    parser = argparse.ArgumentParser(description="ARAG Batch Runner")
    parser.add_argument("--config", "-c", required=True, help="Config file path")
    parser.add_argument("--questions", "-q", required=True, help="Questions file path")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--workers", "-w", type=int, default=10, help="Number of workers")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    runner = BatchRunner(
        config=config,
        questions_file=args.questions,
        output_dir=args.output,
        limit=args.limit,
        num_workers=args.workers,
        verbose=args.verbose
    )

    runner.run()


if __name__ == "__main__":
    main()
