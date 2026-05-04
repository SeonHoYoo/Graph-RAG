"""
Infill 응답 문장과 문서 트리플 간 임베딩 유사도를 계산하는 검증 스크립트
"""

import argparse
import json
import os
import re
from typing import List

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--max_length", type=int, default=256, help="Tokenizer max length.")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip().strip(",").strip('"').strip("'")


def parse_response_sentences(response: str) -> List[str]:
    if not isinstance(response, str):
        return []
    text = response.strip()
    if not text:
        return []

    # 먼저 JSON 파싱을 시도합니다(응답이 JSON 배열 문자열일 수 있음).
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [clean_text(x) for x in parsed if clean_text(x)]
        if isinstance(parsed, str):
            text = parsed
    except Exception:
        pass

    if "\n" in text:
        lines = [clean_text(x) for x in text.splitlines() if clean_text(x)]
        if lines:
            return lines

    # 일반 형식: "triple1", "triple2", ...
    parts = re.split(r'"\s*,\s*"', text)
    if len(parts) > 1:
        parsed_parts = [clean_text(p) for p in parts if clean_text(p)]
        if parsed_parts:
            return parsed_parts

    # 예비 처리: 문장 단위 분리
    fallback = [clean_text(x) for x in re.split(r"(?<=[.!?])\s+", text) if clean_text(x)]
    return fallback


class HFEmbedder:
    def __init__(self, model_name: str, max_length: int = 256):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                "Failed to import embedding dependencies. "
                "Please install/activate an environment with torch + transformers."
            ) from e

        self.torch = torch
        self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.max_length = max_length

    def encode(self, texts: List[str], batch_size: int = 32):
        if not texts:
            return self.torch.empty((0, 0), dtype=self.torch.float32, device=self.device)

        all_embeddings = []
        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)
                token_embeddings = outputs.last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_embeddings.append(pooled)
        return self.torch.cat(all_embeddings, dim=0)


def main():
    args = parse_args()

    with open(args.data_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    if args.output_file is None:
        input_stem = os.path.splitext(os.path.basename(args.data_file))[0]
        output_dir = os.path.dirname(args.data_file)
        args.output_file = os.path.join(output_dir, f"{input_stem}_embedding_top{args.top_k}.json")

    embedder = HFEmbedder(args.model_name, max_length=args.max_length)
    results = []

    for sample in tqdm(samples, desc="Embedding match"):
        index = sample.get("index")
        response = sample.get("infill_result", {}).get("response", "")
        response_sentences = parse_response_sentences(response)
        doc_triples = sample.get("document_graph", {}).get("triples", [])
        if not isinstance(doc_triples, list):
            doc_triples = []
        doc_triples = [clean_text(x) for x in doc_triples if clean_text(x)]

        sample_result = {"index": index, "embedding_matches": []}
        if not response_sentences or not doc_triples:
            results.append(sample_result)
            continue

        response_embeddings = embedder.encode(response_sentences, batch_size=args.batch_size)
        doc_embeddings = embedder.encode(doc_triples, batch_size=args.batch_size)
        similarity = response_embeddings @ doc_embeddings.T

        k = len(doc_triples) if args.top_k <= 0 else min(args.top_k, len(doc_triples))
        top_values, top_indices = embedder.torch.topk(similarity, k=k, dim=1)

        for sent_idx, sentence in enumerate(response_sentences):
            matched_triples = [doc_triples[doc_idx] for doc_idx in top_indices[sent_idx].tolist()]
            matched_scores = [round(float(score), 6) for score in top_values[sent_idx].tolist()]
            sample_result["embedding_matches"].append(
                {
                    "response_sentence": sentence,
                    "top_matches": {
                        "triple": matched_triples,
                        "score": matched_scores,
                    },
                }
            )

        results.append(sample_result)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved: {args.output_file}")
    print(f"Processed samples: {len(results)}")


if __name__ == "__main__":
    main()
