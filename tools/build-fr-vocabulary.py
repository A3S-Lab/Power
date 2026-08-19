#!/usr/bin/env python3
"""Build a deterministic frequency-ranked vocabulary for an MTP draft head.

By default the script follows the FR-Spec calibration recipe and streams
SlimPajama. Local JSONL files can be supplied for offline or domain-specific
calibration. The output is consumed by ``add-gguf-mtp-head.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def select_ranked_token_ids(
    counts: Counter[int],
    special_token_ids: Iterable[int],
    vocabulary_size: int,
    limit: int,
) -> list[int]:
    if limit <= 0 or limit > vocabulary_size:
        raise ValueError(
            f"vocabulary size must be in [1, {vocabulary_size}], got {limit}"
        )

    ranked = sorted(
        (token_id for token_id, count in counts.items() if count > 0),
        key=lambda token_id: (-counts[token_id], token_id),
    )
    if len(ranked) < limit:
        raise ValueError(
            f"corpus exposes only {len(ranked)} unique tokens; "
            f"cannot build a {limit}-token vocabulary"
        )

    special = sorted(set(special_token_ids))
    invalid = next(
        (token_id for token_id in special if token_id < 0 or token_id >= vocabulary_size),
        None,
    )
    if invalid is not None:
        raise ValueError(f"special token ID {invalid} is outside the tokenizer vocabulary")
    if len(special) > limit:
        raise ValueError("FR vocabulary is smaller than the tokenizer special-token set")

    selected = ranked[:limit]
    selected_set = set(selected)
    missing_special = [token_id for token_id in special if token_id not in selected_set]
    protected = set(special)
    replacement_indices = [
        index for index in range(len(selected) - 1, -1, -1)
        if selected[index] not in protected
    ]
    for token_id, index in zip(
        missing_special,
        replacement_indices[: len(missing_special)],
        strict=True,
    ):
        selected_set.remove(selected[index])
        selected[index] = token_id
        selected_set.add(token_id)

    if len(selected_set) != limit:
        raise AssertionError("frequency ranking produced duplicate token IDs")
    return selected


def jsonl_documents(paths: list[Path], text_field: str) -> Iterator[str]:
    for path in paths:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                document = json.loads(line)
                text = document.get(text_field) if isinstance(document, dict) else None
                if not isinstance(text, str):
                    raise ValueError(
                        f"{path}:{line_number} has no string field {text_field!r}"
                    )
                yield text


def dataset_documents(
    dataset_name: str,
    dataset_config: str,
    split: str,
    revision: str,
    text_field: str,
) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face datasets is required for streaming calibration"
        ) from error

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        revision=revision,
        streaming=True,
    )
    for document in dataset:
        text = document.get(text_field)
        if isinstance(text, str):
            yield text


def resolve_hugging_face_revision(repository: str, revision: str, repo_type: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    if repo_type == "model":
        return api.model_info(repository, revision=revision).sha
    return api.dataset_info(repository, revision=revision).sha


def count_tokens(
    documents: Iterable[str],
    tokenizer: Any,
    max_documents: int,
) -> tuple[Counter[int], int, int]:
    counts: Counter[int] = Counter()
    document_count = 0
    token_count = 0
    for text in documents:
        if document_count >= max_documents:
            break
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        counts.update(token_ids)
        document_count += 1
        token_count += len(token_ids)
        if document_count % 1_000 == 0:
            print(
                f"Processed {document_count}/{max_documents} documents "
                f"({token_count} tokens, {len(counts)} unique)",
                flush=True,
            )
    return counts, document_count, token_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output JSON vocabulary file")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--tokenizer-revision", default="main")
    parser.add_argument("--dataset", default="DKYoon/SlimPajama-6B")
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--input-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-documents", type=int, default=1_000_000)
    parser.add_argument("--vocab-size", type=int, default=65_536)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.max_documents <= 0:
        raise ValueError("--max-documents must be positive")

    from transformers import AutoTokenizer

    tokenizer_revision = resolve_hugging_face_revision(
        args.tokenizer,
        args.tokenizer_revision,
        "model",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=tokenizer_revision,
    )

    dataset_revision: str | None = None
    if args.input_jsonl:
        documents = jsonl_documents(args.input_jsonl, args.text_field)
        corpus = [str(path.resolve()) for path in args.input_jsonl]
    else:
        dataset_revision = resolve_hugging_face_revision(
            args.dataset,
            args.dataset_revision,
            "dataset",
        )
        documents = dataset_documents(
            args.dataset,
            args.dataset_config,
            args.dataset_split,
            dataset_revision,
            args.text_field,
        )
        corpus = args.dataset

    counts, document_count, token_count = count_tokens(
        documents,
        tokenizer,
        args.max_documents,
    )
    if document_count != args.max_documents:
        raise ValueError(
            f"calibration source ended after {document_count} documents; "
            f"expected {args.max_documents}"
        )
    token_ids = select_ranked_token_ids(
        counts,
        tokenizer.all_special_ids,
        len(tokenizer),
        args.vocab_size,
    )

    output = {
        "schema_version": 1,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_vocabulary_size": len(tokenizer),
        "corpus": corpus,
        "dataset_revision": dataset_revision,
        "documents": document_count,
        "tokens": token_count,
        "token_ids": token_ids,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as destination:
        json.dump(output, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(f"Wrote {len(token_ids)} ranked token IDs to {args.output}", flush=True)


if __name__ == "__main__":
    main()
