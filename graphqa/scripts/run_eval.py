"""TASI 전체 평가 진입점 스크립트.

사용 예 (LLM 미사용):
  python -m graphqa.scripts.run_eval \
        --datasets 2wikimultihopqa hotpotqa musique \
        --output-dir graphqa/outputs/full --save-plots

사용 예 (Qwen2.5-7B 로컬):
  python -m graphqa.scripts.run_eval \
        --datasets 2wikimultihopqa hotpotqa musique \
        --output-dir graphqa/outputs/full_llm \
        --use-llm qwen-local --llm-model Qwen/Qwen2.5-7B-Instruct \
        --save-plots
"""
from __future__ import annotations

import argparse
import logging
import sys
import pathlib
from typing import List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd

from graphqa.data import DATASETS, load_dataset
from graphqa.evaluate import (
    DEFAULT_SWEEP_THRESHOLDS,
    SCORE_COLS,
    evaluate_all,
    print_selective_report,
    print_summary,
    selective_report,
    _summarize,
)
from graphqa.llm_qa import LLMBackend, make_backend
from graphqa.pipeline import TASIPipeline
from graphqa.tasi.embedding import get_default_encoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                   choices=list(DATASETS.keys()))
    p.add_argument("--limit", type=int, default=None,
                   help="각 데이터셋 최대 샘플 수")
    p.add_argument("--output-dir", type=str, default="graphqa/outputs/full")
    p.add_argument("--triplets-filename", type=str, default=None,
                   help="results/{dataset}/triplets/{model_dir}/ 아래에서 읽을 "
                        "Q/D triplets 파일명. 예: "
                        "triplets_train_sampled_open-book_top10.json")
    p.add_argument("--triplets-model-dir", type=str,
                   default="Qwen2.5-7B-Instruct",
                   help="triplets 파일이 있는 model directory 이름.")
    p.add_argument("--triplets-file", type=str, default=None,
                   help="단일 dataset 실행 시 사용할 triplets JSON 절대/상대 경로.")
    p.add_argument("--encoder", type=str,
                   default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--use-hungarian", action="store_true")
    p.add_argument("--no-inverse", action="store_true")
    p.add_argument("--no-step-pc", action="store_true")
    p.add_argument("--save-plots", action="store_true")

    # PC behavior
    p.add_argument("--pc-mode", type=str, default="log_mean",
                   choices=["product", "log_mean", "mean"],
                   help="propagation consistency 결합 방식")
    p.add_argument("--pc-soft", action="store_true", default=True,
                   help="토큰 단위 soft overlap (기본 True)")
    p.add_argument("--no-pc-soft", dest="pc_soft", action="store_false")
    p.add_argument("--pc-epsilon", type=float, default=0.05)

    # LLM
    p.add_argument("--use-llm", type=str, default="none",
                   choices=["none", "dummy", "qwen-local", "qwen", "openai"])
    p.add_argument("--llm-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--llm-device", type=str, default=None)
    p.add_argument("--llm-dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--openai-key", type=str, default=None)
    p.add_argument("--llm-prompt-mode", type=str, default="extract",
                   choices=["extract", "reason"],
                   help="extract: chain entity 중 답 고르기 (현재 기본). "
                        "reason: chain·evidence 를 reasoning 단서로만 쓰고 LLM 자유 추론.")

    # Verifier (ensemble + TASI grounding + abstain)
    p.add_argument("--use-verifier", action="store_true",
                   help="extract+reason+TASI 후보를 TASI 그래프 기반 verifier 로 re-rank.")
    p.add_argument("--abstain-threshold", type=float, default=0.0,
                   help="verify_score 가 이 값 미만이면 abstain 처리 (fallback to extract LLM).")
    p.add_argument("--verifier-no-extract", dest="verifier_use_extract",
                   action="store_false", default=True)
    p.add_argument("--verifier-no-reason", dest="verifier_use_reason",
                   action="store_false", default=True)
    p.add_argument("--verifier-no-tasi", dest="verifier_use_tasi",
                   action="store_false", default=True)
    p.add_argument("--verifier-w-grounding", type=float, default=0.5)
    p.add_argument("--verifier-w-chain", type=float, default=0.3)
    p.add_argument("--verifier-w-type", type=float, default=0.2)

    # Selective prediction sweep (post-eval, no extra LLM calls)
    p.add_argument("--abstain-thresholds", type=float, nargs="+", default=None,
                   help="threshold sweep 값 list. 미지정 시 기본 sweep grid.")

    # QA mode: 'auto' (verifier 사용 여부에 따라), 'verifier', 'tasi_gated', 'evidence', 'iterative'
    p.add_argument("--qa-mode", type=str, default="auto",
                   choices=["auto", "verifier", "tasi_gated", "evidence",
                            "single", "iterative", "triplet_fill"],
                   help="auto: --use-verifier flag 따라 verifier 또는 single. "
                        "tasi_gated: TASI top-K candidate + pre/post gate 사용. "
                        "evidence: TASI 미사용. LLM 이 chain + cosine top-K evidence "
                        "만 보고 직접 infill + 답. "
                        "iterative: UNKNOWN 슬롯을 랜덤 순서로 한 개씩 LLM 채움 후 최종 답. "
                        "triplet_fill: query triple을 doc triple field 검증으로 순차 채움.")
    p.add_argument("--iter-abstain", action="store_true",
                   help="iterative 전용: 스텝마다 alignment 신호 주입 + 중간 abstain 허용 "
                        "(비교군). 미지정 시 base(신호 없음, 중간 abstain 없음).")
    p.add_argument("--triplet-fill-threshold", type=float, default=0.50,
                   help="triplet_fill: field-level min cosine 통과 임계값.")
    p.add_argument("--triplet-fill-max-steps", type=int, default=16,
                   help="triplet_fill: 한 샘플에서 시도할 최대 query triple 채움 횟수.")
    p.add_argument("--triplet-fill-doc-top-k", type=int, default=1,
                   help="triplet_fill: document 후보를 whole-triple cosine top-K로 "
                        "좁힌 뒤 field-level score 최고 후보를 선택.")
    p.add_argument("--triplet-fill-answer-on-fail", action="store_true",
                   help="triplet_fill: doc alignment 실패/미채움/step 초과 시에도 "
                        "현재까지 채운 partial chain으로 최종 LLM 답변을 생성.")
    p.add_argument("--triplet-fill-think-rescue", action="store_true",
                   help="triplet_fill: doc alignment 실패 + think validation 통과 시 "
                        "think triple의 concrete 슬롯값으로 채우고 계속 진행.")
    p.add_argument("--combined-dir", type=str, default=None,
                   help="combined/0514 step별 doc/think evidence JSON 디렉터리.")
    p.add_argument("--triplet-fill-evidence-scope", type=str,
                   default="legacy_full",
                   choices=[
                       "legacy_full",
                       "combined_full",
                       "combined_strict",
                       "combined_prefix",
                   ],
                   help="triplet_fill evidence pool: legacy 전체 D/T 또는 "
                        "combined step evidence full/strict/prefix.")
    p.add_argument("--gated-k", type=int, default=5,
                   help="tasi_gated: 슬롯당 top-K 후보 entity 수.")
    p.add_argument("--gated-pre-threshold", type=float, default=0.0,
                   help="tasi_gated: total_sum < τ 면 LLM 호출 없이 abstain.")
    p.add_argument("--gated-post-uplift-min", type=float, default=-0.05,
                   help="tasi_gated: 채워진 chain 의 alignment uplift 가 이 값 미만이면 abstain.")
    p.add_argument("--gated-no-pre", dest="gated_enable_pre",
                   action="store_false", default=True)
    p.add_argument("--gated-no-post", dest="gated_enable_post",
                   action="store_false", default=True)

    p.add_argument("--inject-alignment-signal", action="store_true",
                   help="tasi_gated: prompt 에 (Q,D)/(Q,Sr)/(Q,T) sentence-cosine "
                        "alignment 점수를 'Alignment signal' 섹션으로 주입.")

    return p.parse_args()


def make_llm(args: argparse.Namespace) -> Optional[LLMBackend]:
    name = (args.use_llm or "none").lower()
    if name in ("none", "off"):
        return None
    if name in ("qwen-local", "qwen"):
        return make_backend("qwen-local", model_name=args.llm_model,
                            device=args.llm_device, torch_dtype=args.llm_dtype)
    if name == "openai":
        return make_backend("openai", model=args.llm_model, api_key=args.openai_key)
    if name == "dummy":
        return make_backend("dummy")
    return None


def main() -> int:
    args = parse_args()
    if args.triplets_file and len(args.datasets) != 1:
        raise SystemExit("--triplets-file can only be used with exactly one dataset")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

    encoder = get_default_encoder(model_name=args.encoder, device=args.device)
    pipeline = TASIPipeline(
        encoder=encoder,
        use_hungarian=args.use_hungarian,
        allow_inverse=not args.no_inverse,
        use_steps_for_T=not args.no_step_pc,
        pc_mode=args.pc_mode,
        pc_soft=args.pc_soft,
        pc_epsilon=args.pc_epsilon,
    )
    llm_backend = make_llm(args)
    qm = (args.qa_mode or "auto").lower()
    if qm in ("iterative", "triplet_fill") and llm_backend is None:
        from graphqa.llm_qa import DummyBackend
        llm_backend = DummyBackend()
        logging.info(f"qa-mode={qm}: LLM 미지정 → DummyBackend (스모크)")
    if llm_backend is not None:
        logging.info(f"LLM backend: {llm_backend.name}")

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs: List[pd.DataFrame] = []

    for ds in args.datasets:
        print("\n" + "#" * 90)
        print(f"# dataset = {ds}")
        print("#" * 90)
        combined_dir = (
            args.combined_dir
            if str(args.triplet_fill_evidence_scope).startswith("combined")
            else None
        )
        if str(args.triplet_fill_evidence_scope).startswith("combined") and not combined_dir:
            raise SystemExit(
                "--combined-dir is required for combined_* triplet-fill evidence scopes"
            )
        samples = load_dataset(
            ds,
            limit=args.limit,
            combined_dir=combined_dir,
            triplets_file=args.triplets_file,
            triplets_filename=args.triplets_filename,
            triplets_model_dir=args.triplets_model_dir,
        )
        if not samples:
            logging.warning(f"no samples for {ds}, skipping")
            continue

        verifier_weights = {
            "grounding": args.verifier_w_grounding,
            "chain": args.verifier_w_chain,
            "type": args.verifier_w_type,
        }
        df, summary = evaluate_all(
            samples,
            pipeline=pipeline,
            llm_backend=llm_backend,
            output_dir=output_dir / ds,
            output_name=f"tasi_eval_{ds}",
            save_plots=args.save_plots,
            prompt_mode=args.llm_prompt_mode,
            use_verifier=args.use_verifier,
            verifier_weights=verifier_weights,
            abstain_threshold=args.abstain_threshold,
            verifier_use_extract=args.verifier_use_extract,
            verifier_use_reason=args.verifier_use_reason,
            verifier_use_tasi=args.verifier_use_tasi,
            qa_mode=args.qa_mode,
            gated_k_per_slot=args.gated_k,
            gated_pre_threshold=args.gated_pre_threshold,
            gated_post_uplift_min=args.gated_post_uplift_min,
            gated_enable_pre=args.gated_enable_pre,
            gated_enable_post=args.gated_enable_post,
            inject_alignment_signal=args.inject_alignment_signal,
            iter_abstain=args.iter_abstain,
            triplet_fill_threshold=args.triplet_fill_threshold,
            triplet_fill_max_steps=args.triplet_fill_max_steps,
            triplet_fill_answer_on_fail=args.triplet_fill_answer_on_fail,
            triplet_fill_doc_top_k=args.triplet_fill_doc_top_k,
            triplet_fill_think_rescue=args.triplet_fill_think_rescue,
            triplet_fill_evidence_scope=args.triplet_fill_evidence_scope,
        )
        print_summary(summary)

        # 사용자가 sweep grid 를 명시했다면 한 번 더 그 grid 로 출력 + JSON 갱신
        if args.use_verifier and args.abstain_thresholds is not None \
                and "verify_score" in df.columns and df["verify_score"].notna().any():
            user_report = selective_report(df, thresholds=args.abstain_thresholds)
            print(f"\n[user-grid sweep — {ds}]")
            print_selective_report(user_report)
            user_json = output_dir / ds / f"tasi_eval_{ds}_sweep.json"
            try:
                user_json.parent.mkdir(parents=True, exist_ok=True)
                import json as _json
                with open(user_json, "w", encoding="utf-8") as f:
                    _json.dump(user_report, f, indent=2, ensure_ascii=False)
                logging.info(f"saved user-grid sweep: {user_json}")
            except Exception as exc:
                logging.warning(f"sweep write failed for {ds}: {exc}")

        all_dfs.append(df)

    if len(all_dfs) > 1:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = output_dir / "tasi_eval_all.csv"
        combined.to_csv(combined_path, index=False)
        logging.info(f"saved combined CSV: {combined_path}")

        all_summary = _summarize(combined)
        print("\n" + "=" * 90)
        print("# Combined across all datasets")
        print("=" * 90)
        print_summary(all_summary)

        if args.use_verifier and "verify_score" in combined.columns \
                and combined["verify_score"].notna().any():
            grid = args.abstain_thresholds or list(DEFAULT_SWEEP_THRESHOLDS)
            combined_report = selective_report(combined, thresholds=grid)
            print("\n[combined sweep — all datasets]")
            print_selective_report(combined_report)
            comb_json = output_dir / "tasi_eval_all_sweep.json"
            try:
                import json as _json
                with open(comb_json, "w", encoding="utf-8") as f:
                    _json.dump(combined_report, f, indent=2, ensure_ascii=False)
                logging.info(f"saved combined sweep: {comb_json}")
            except Exception as exc:
                logging.warning(f"combined sweep write failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
