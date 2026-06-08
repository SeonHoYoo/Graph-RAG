"""Search-R1 reasoning step을 triplet으로 쪼개 reasoning_graph JSON을 만드는 스크립트.

입력: graph_data/searchr1/<setting>/*.json
     (각 sample은 `reasoning_steps`(list[str]) 또는
      `retrieval_info.searchr1_reasoning_steps`를 포함)
출력: 동일한 sample 메타데이터 + `reasoning_graph` 필드
       - reasoning_graph.definition_triples: 전체 합친 latent entity 정의
       - reasoning_graph.triples:            전체 합친 triple 리스트
       - reasoning_graph.per_step:           step별 triple 결과
"""
import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_library.construct_model import ConstructModel


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, required=True,
        help="Search-R1 reasoning JSON 파일 경로")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None,
        choices=[None, "musique", "hotpotqa", "2wikimultihopqa"],
        help="생략 시 파일 경로에서 추론")
    parser.add_argument("--mode", type=str, default="per_step",
        choices=["per_step", "whole"],
        help="per_step: 각 reasoning step마다 LLM 호출 / whole: 전체 reasoning_path 한 번에 호출")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_filename", type=str, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=10,
        help="N개 처리될 때마다 중간 저장")
    return parser.parse_args()


def infer_dataset(input_path: str) -> str:
    path_lower = input_path.lower()
    if "2wiki" in path_lower or "2wikimultihopqa" in path_lower:
        return "2wikimultihopqa"
    if "hotpotqa" in path_lower or "hotpot" in path_lower:
        return "hotpotqa"
    if "musique" in path_lower:
        return "musique"
    raise ValueError(
        "데이터셋을 파일 경로에서 추론할 수 없습니다. --dataset 인자를 직접 지정해 주세요."
    )


def get_reasoning_steps(sample: Dict[str, Any]) -> List[str]:
    steps = sample.get("reasoning_steps")
    if steps:
        return steps
    retrieval_info = sample.get("retrieval_info", {}) or {}
    steps = retrieval_info.get("searchr1_reasoning_steps") or []
    return steps


def get_reasoning_path(sample: Dict[str, Any]) -> str:
    path = sample.get("reasoning_path")
    if path:
        return path
    retrieval_info = sample.get("retrieval_info", {}) or {}
    path = retrieval_info.get("searchr1_reasoning_path") or ""
    if not path:
        path = "\n".join(get_reasoning_steps(sample))
    return path


def extract_per_step(
    construct_model: ConstructModel,
    reasoning_steps: List[str],
) -> Dict[str, Any]:
    per_step: List[Dict[str, Any]] = []
    all_def: List[str] = []
    all_triples: List[str] = []
    for step_idx, step_text in enumerate(reasoning_steps):
        step_text = (step_text or "").strip()
        if not step_text:
            per_step.append({
                "step_index": step_idx,
                "step_text": step_text,
                "definition_triples": [],
                "triples": [],
            })
            continue
        try:
            def_t, tri = construct_model.extract_triplets_from_cot_reasoning(step_text)
        except Exception as exc:
            logger.warning("step %d triplet extraction failed: %s", step_idx, exc)
            def_t, tri = [], []
        per_step.append({
            "step_index": step_idx,
            "step_text": step_text,
            "definition_triples": def_t,
            "triples": tri,
        })
        all_def.extend(def_t)
        all_triples.extend(tri)

    def _dedup(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return {
        "definition_triples": _dedup(all_def),
        "triples": _dedup(all_triples),
        "per_step": per_step,
    }


def extract_whole(
    construct_model: ConstructModel,
    reasoning_path: str,
) -> Dict[str, Any]:
    reasoning_path = (reasoning_path or "").strip()
    if not reasoning_path:
        return {"definition_triples": [], "triples": [], "per_step": []}
    try:
        def_t, tri = construct_model.extract_triplets_from_cot_reasoning(reasoning_path)
    except Exception as exc:
        logger.warning("whole reasoning triplet extraction failed: %s", exc)
        def_t, tri = [], []
    return {"definition_triples": def_t, "triples": tri, "per_step": []}


def build_result(
    sample: Dict[str, Any],
    reasoning_steps: List[str],
    reasoning_graph: Dict[str, Any],
) -> Dict[str, Any]:
    retrieval_info = sample.get("retrieval_info", {}) or {}
    return {
        "index": sample.get("index"),
        "uid": sample.get("uid"),
        "num_hops": sample.get("num_hops"),
        "question": sample.get("question", ""),
        "answer": sample.get("answer"),
        "answer_aliases": sample.get("answer_aliases", []),
        "gold_id_list": sample.get("gold_id_list", []),
        "predicted_answer": sample.get("predicted_answer"),
        "answer_matches_gold": sample.get("answer_matches_gold"),
        "reasoning_steps": reasoning_steps,
        "reasoning_path": sample.get("reasoning_path")
            or retrieval_info.get("searchr1_reasoning_path", ""),
        "reasoning_graph": reasoning_graph,
    }


def main() -> None:
    args = parse_args()
    input_path = os.path.abspath(args.input_file_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    dataset = args.dataset or infer_dataset(input_path)
    logger.info("Inferred dataset = %s", dataset)

    construct_model = ConstructModel(
        construct_model_name=args.model_name,
        dataset_name=dataset,
        api_key=args.api_key,
        batch_size=1,
    )

    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    logger.info("Loaded %d samples from %s", len(samples), input_path)

    if args.max_samples is not None:
        samples = samples[: args.max_samples]
        logger.info("Limiting to %d samples", len(samples))

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = args.output_filename or (
        f"reasoning_graph_{os.path.splitext(os.path.basename(input_path))[0]}.json"
    )
    output_path = os.path.join(args.output_dir, output_name)

    results: List[Dict[str, Any]] = []
    done_indices = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            done_indices = {r.get("index") for r in results if r.get("index") is not None}
            logger.info("Resuming from checkpoint: %d samples already processed", len(done_indices))
        except Exception as exc:
            logger.warning("Failed to load existing checkpoint (%s). Starting fresh.", exc)
            results = []
            done_indices = set()

    new_count = 0
    for sample in tqdm(samples, desc="Extracting reasoning graphs"):
        sample_idx = sample.get("index")
        if sample_idx is not None and sample_idx in done_indices:
            continue

        reasoning_steps = get_reasoning_steps(sample)
        try:
            if args.mode == "whole":
                reasoning_path = get_reasoning_path(sample)
                reasoning_graph = extract_whole(construct_model, reasoning_path)
            else:
                reasoning_graph = extract_per_step(construct_model, reasoning_steps)
        except Exception as exc:
            logger.error("Failed on index=%s: %s", sample_idx, exc)
            reasoning_graph = {"definition_triples": [], "triples": [], "per_step": []}

        results.append(build_result(sample, reasoning_steps, reasoning_graph))
        new_count += 1

        if args.checkpoint_every > 0 and new_count % args.checkpoint_every == 0:
            results.sort(key=lambda r: (r.get("index") is None, r.get("index")))
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info("Checkpoint saved (%d samples) -> %s", len(results), output_path)

    results.sort(key=lambda r: (r.get("index") is None, r.get("index")))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", output_path)

    triple_counts = [len(r["reasoning_graph"].get("triples", [])) for r in results]
    if triple_counts:
        logger.info(
            "Avg triples per sample: %.2f (min=%d, max=%d, total=%d)",
            sum(triple_counts) / len(triple_counts),
            min(triple_counts),
            max(triple_counts),
            sum(triple_counts),
        )


if __name__ == "__main__":
    main()
