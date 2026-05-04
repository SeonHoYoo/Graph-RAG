import argparse
import json
import os
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract hidden-state snapshots immediately before generated step-boundary markers."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--boundary",
        action="append",
        dest="boundaries",
        default=[],
        help="Literal decoded text boundary marker. Can be passed multiple times.",
    )
    parser.add_argument(
        "--layer",
        action="append",
        dest="layers",
        type=int,
        default=None,
        help="Layer indices to save. Defaults to final layer only.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Storage dtype for saved tensors.",
    )
    return parser.parse_args()


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    raise ValueError("Either --prompt or --prompt_file must be provided.")


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def normalize_boundaries(boundaries: List[str]) -> List[str]:
    if boundaries:
        return boundaries
    return ["Step 1:", "Step 2:", "Step 3:", "\n\n"]


def token_count(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def find_completed_boundary(text: str, boundaries: List[str]) -> Optional[str]:
    for boundary in sorted(boundaries, key=len, reverse=True):
        if text.endswith(boundary):
            return boundary
    return None


def summarize_vector(tensor: torch.Tensor) -> Dict[str, float]:
    flat = tensor.float().cpu()
    return {
        "l2_norm": torch.norm(flat, p=2).item(),
        "mean": flat.mean().item(),
        "std": flat.std().item(),
        "max_abs": flat.abs().max().item(),
    }


def main() -> None:
    args = parse_args()
    prompt = load_prompt(args)
    boundaries = normalize_boundaries(args.boundaries)
    save_dtype = resolve_dtype(args.dtype)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype="auto",
        device_map="auto",
    )
    model.eval()

    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]

    eos_token_id = tokenizer.eos_token_id
    snapshots = []
    captured_token_spans = set()

    with torch.no_grad():
        for step_idx in range(args.max_new_tokens):
            outputs = model(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            next_token_logits = outputs.logits[:, -1, :]
            if args.temperature > 0:
                probs = torch.softmax(next_token_logits / args.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)],
                dim=-1,
            )

            generated_text = tokenizer.decode(
                generated_ids[0][model_inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            matched_boundary = find_completed_boundary(generated_text, boundaries)
            if matched_boundary is not None:
                boundary_token_len = token_count(tokenizer, matched_boundary)
                generated_token_len = generated_ids.shape[1] - model_inputs["input_ids"].shape[1]
                boundary_end = generated_token_len
                boundary_start = boundary_end - boundary_token_len
                capture_key = (boundary_start, boundary_end, matched_boundary)

                if boundary_start > 0 and capture_key not in captured_token_spans:
                    last_hidden_states = outputs.hidden_states
                    available_layers = list(range(len(last_hidden_states)))
                    target_layers = args.layers if args.layers is not None else [len(last_hidden_states) - 1]

                    layer_vectors = {}
                    layer_summaries = {}
                    seq_index = model_inputs["input_ids"].shape[1] + boundary_start - 1

                    for layer_idx in target_layers:
                        if layer_idx not in available_layers:
                            raise ValueError(f"Requested layer {layer_idx} is out of range 0..{len(last_hidden_states)-1}")
                        vector = last_hidden_states[layer_idx][0, seq_index, :].detach().to("cpu", dtype=save_dtype)
                        layer_vectors[str(layer_idx)] = vector
                        layer_summaries[str(layer_idx)] = summarize_vector(vector)

                    snapshot_idx = len(snapshots)
                    tensor_path = os.path.join(args.output_dir, f"snapshot_{snapshot_idx:03d}.pt")
                    torch.save(
                        {
                            "boundary": matched_boundary,
                            "boundary_start_generated_token": boundary_start,
                            "boundary_end_generated_token": boundary_end,
                            "generated_text": generated_text,
                            "prompt": prompt,
                            "layers": layer_vectors,
                        },
                        tensor_path,
                    )

                    snapshots.append(
                        {
                            "snapshot_index": snapshot_idx,
                            "boundary": matched_boundary,
                            "step_index": step_idx,
                            "boundary_start_generated_token": boundary_start,
                            "boundary_end_generated_token": boundary_end,
                            "decoded_prefix_before_boundary": tokenizer.decode(
                                generated_ids[0][model_inputs["input_ids"].shape[1]: model_inputs["input_ids"].shape[1] + boundary_start],
                                skip_special_tokens=True,
                            ),
                            "decoded_text_through_boundary": generated_text,
                            "tensor_path": tensor_path,
                            "layer_summaries": layer_summaries,
                        }
                    )
                    captured_token_spans.add(capture_key)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    metadata = {
        "model_name": args.model_name,
        "prompt": prompt,
        "boundaries": boundaries,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "num_snapshots": len(snapshots),
        "snapshots": snapshots,
        "final_text": tokenizer.decode(
            generated_ids[0][model_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ),
    }

    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(json.dumps({"metadata_path": metadata_path, "num_snapshots": len(snapshots)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
