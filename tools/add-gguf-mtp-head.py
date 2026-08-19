#!/usr/bin/env python3
"""Add a separately quantized MTP output head to a GGUF model.

The base model is copied without changing any existing tensor. A selected output
tensor is read from a second GGUF and appended under the Qwen NextN head name,
allowing the target context to keep its original output projection while an MTP
draft context uses the cheaper projection. An optional frequency-ranked token
list compacts the draft head and adds a draft-to-target ID map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gguf
except ImportError as error:
    raise SystemExit(
        "The llama.cpp gguf Python package is required; add gguf-py to PYTHONPATH"
    ) from error


SOURCE_HEAD_NAME = "output.weight"
MTP_HEAD_NAME = "blk.64.nextn.shared_head_head.weight"
DRAFT_TO_TARGET_NAME = "d2t"
FR_MAPPING_KEY = "a3s.mtp.fr.mapping"
FR_VOCABULARY_SIZE_KEY = "a3s.mtp.fr.vocabulary_size"
FR_D2T_SHA256_KEY = "a3s.mtp.fr.d2t_sha256"
FR_METADATA_KEYS = frozenset(
    {FR_MAPPING_KEY, FR_VOCABULARY_SIZE_KEY, FR_D2T_SHA256_KEY}
)


def field_value(reader: gguf.GGUFReader, key: str) -> Any:
    field = reader.get_field(key)
    return field.contents() if field else None


def copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for field in reader.fields.values():
        if (
            field.name == gguf.Keys.General.ARCHITECTURE
            or field.name.startswith("GGUF.")
            or field.name in FR_METADATA_KEYS
        ):
            continue

        value_type = field.types[0]
        subtype = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(
            field.name,
            field.contents(),
            value_type,
            sub_type=subtype,
        )


def tensor_by_name(reader: gguf.GGUFReader, name: str) -> gguf.ReaderTensor:
    tensor = next((candidate for candidate in reader.tensors if candidate.name == name), None)
    if tensor is None:
        raise ValueError(f"GGUF does not contain tensor {name!r}")
    return tensor


def add_tensor_info(writer: gguf.GGUFWriter, tensor: gguf.ReaderTensor, name: str) -> None:
    writer.add_tensor_info(
        name,
        tensor.data.shape,
        tensor.data.dtype,
        tensor.data.nbytes,
        tensor.tensor_type,
    )


def load_frequency_token_ids(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as source:
        document = json.load(source)

    values = document.get("token_ids") if isinstance(document, dict) else document
    if not isinstance(values, list):
        raise ValueError("frequency token file must be a JSON list or contain token_ids")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("frequency token IDs must be integers")
    return values


def validate_frequency_token_ids(
    token_ids: list[int],
    vocabulary_size: int,
    limit: int | None,
) -> np.ndarray:
    selected_count = len(token_ids) if limit is None else limit
    if selected_count <= 0:
        raise ValueError("FR vocabulary size must be positive")
    if selected_count > len(token_ids):
        raise ValueError(
            f"FR vocabulary size {selected_count} exceeds the "
            f"{len(token_ids)} available token IDs"
        )

    selected = token_ids[:selected_count]
    if len(set(selected)) != len(selected):
        raise ValueError("frequency token IDs must be unique")
    invalid = next(
        (token_id for token_id in selected if token_id < 0 or token_id >= vocabulary_size),
        None,
    )
    if invalid is not None:
        raise ValueError(
            f"frequency token ID {invalid} is outside [0, {vocabulary_size})"
        )
    return np.asarray(selected, dtype=np.int64)


def select_frequency_ranked_rows(
    data: np.ndarray[Any, Any],
    token_ids: np.ndarray[Any, Any],
    vocabulary_size: int,
) -> np.ndarray[Any, Any]:
    if data.ndim != 2 or data.shape[0] != vocabulary_size:
        raise ValueError(
            "output tensor storage is not row-addressable by vocabulary token"
        )
    return np.ascontiguousarray(data[token_ids])


def tensors_to_copy(
    tensors: list[gguf.ReaderTensor],
    replace_existing_head: bool,
) -> list[gguf.ReaderTensor]:
    if not replace_existing_head:
        return tensors
    replaced_names = {MTP_HEAD_NAME, DRAFT_TO_TARGET_NAME}
    return [tensor for tensor in tensors if tensor.name not in replaced_names]


def build_model(
    base_path: Path,
    head_path: Path,
    output_path: Path,
    frequency_token_path: Path | None = None,
    fr_vocabulary_size: int | None = None,
    source_head_name: str = SOURCE_HEAD_NAME,
    replace_existing_head: bool = False,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    partial_path = output_path.with_name(f"{output_path.name}.partial")
    if partial_path.exists():
        raise FileExistsError(f"refusing to overwrite {partial_path}")

    base = gguf.GGUFReader(base_path, "r")
    head_source = gguf.GGUFReader(head_path, "r")

    if not replace_existing_head and any(
        tensor.name == MTP_HEAD_NAME for tensor in base.tensors
    ):
        raise ValueError(f"base GGUF already contains {MTP_HEAD_NAME!r}")
    if not replace_existing_head and any(
        tensor.name == DRAFT_TO_TARGET_NAME for tensor in base.tensors
    ):
        raise ValueError(f"base GGUF already contains {DRAFT_TO_TARGET_NAME!r}")
    if fr_vocabulary_size is not None and frequency_token_path is None:
        raise ValueError("--fr-vocab-size requires --frequency-token-ids")

    base_head = tensor_by_name(base, SOURCE_HEAD_NAME)
    draft_head = tensor_by_name(head_source, source_head_name)
    if tuple(base_head.shape) != tuple(draft_head.shape):
        raise ValueError(
            f"head shape mismatch: base={tuple(base_head.shape)}, "
            f"draft={tuple(draft_head.shape)}"
        )
    if base_head.tensor_type == draft_head.tensor_type:
        raise ValueError(
            "draft head has the same quantization type as the base output tensor"
        )

    compact_head: np.ndarray[Any, Any] | None = None
    draft_to_target: np.ndarray[Any, Any] | None = None
    if frequency_token_path is not None:
        if len(draft_head.shape) != 2:
            raise ValueError(f"expected a 2D output tensor, got {draft_head.shape}")
        vocabulary_size = int(draft_head.shape[1])
        token_ids = load_frequency_token_ids(frequency_token_path)
        draft_to_target = validate_frequency_token_ids(
            token_ids,
            vocabulary_size,
            fr_vocabulary_size,
        )
        compact_head = select_frequency_ranked_rows(
            draft_head.data,
            draft_to_target,
            vocabulary_size,
        )

    architecture = field_value(base, gguf.Keys.General.ARCHITECTURE)
    if not isinstance(architecture, str):
        raise ValueError("base GGUF has no valid general.architecture")

    writer = gguf.GGUFWriter(partial_path, arch=architecture, endianess=base.endianess)
    alignment = field_value(base, gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment

    copy_metadata(base, writer)
    if draft_to_target is not None:
        canonical_ids = np.asarray(draft_to_target, dtype="<i8")
        writer.add_string(FR_MAPPING_KEY, "frequency_ranked")
        writer.add_uint32(FR_VOCABULARY_SIZE_KEY, int(draft_to_target.size))
        writer.add_string(
            FR_D2T_SHA256_KEY,
            hashlib.sha256(canonical_ids.tobytes()).hexdigest(),
        )
    copied_tensors = tensors_to_copy(base.tensors, replace_existing_head)
    for tensor in copied_tensors:
        add_tensor_info(writer, tensor, tensor.name)
    if compact_head is None:
        add_tensor_info(writer, draft_head, MTP_HEAD_NAME)
    else:
        writer.add_tensor_info(
            MTP_HEAD_NAME,
            compact_head.shape,
            compact_head.dtype,
            compact_head.nbytes,
            draft_head.tensor_type,
        )
        assert draft_to_target is not None
        writer.add_tensor_info(
            DRAFT_TO_TARGET_NAME,
            draft_to_target.shape,
            draft_to_target.dtype,
            draft_to_target.nbytes,
        )

    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()

        total = len(copied_tensors) + 1 + (1 if draft_to_target is not None else 0)
        for index, tensor in enumerate(copied_tensors, start=1):
            writer.write_tensor_data(tensor.data, tensor_endianess=base.endianess)
            if index % 100 == 0:
                print(f"Copied {index}/{total} tensors", flush=True)

        writer.write_tensor_data(
            draft_head.data if compact_head is None else compact_head,
            tensor_endianess=head_source.endianess,
        )
        if draft_to_target is not None:
            writer.write_tensor_data(draft_to_target)
        writer.close()
        os.replace(partial_path, output_path)
    except BaseException:
        writer.close()
        partial_path.unlink(missing_ok=True)
        raise

    detail = draft_head.tensor_type.name
    if draft_to_target is not None:
        detail += f", frequency-ranked {draft_to_target.size}-token vocabulary"
    print(f"Added {MTP_HEAD_NAME} as {detail}; wrote {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="Original target GGUF")
    parser.add_argument("head_source", type=Path, help="GGUF containing the quantized draft head")
    parser.add_argument("output", type=Path, help="Output GGUF with a separate MTP head")
    parser.add_argument(
        "--source-head-name",
        default=SOURCE_HEAD_NAME,
        help=f"Tensor to copy from head_source (default: {SOURCE_HEAD_NAME})",
    )
    parser.add_argument(
        "--replace-existing-head",
        action="store_true",
        help="Replace an existing MTP head and d2t tensor in the base GGUF",
    )
    parser.add_argument(
        "--frequency-token-ids",
        type=Path,
        help="JSON frequency-ranked token IDs used to compact the MTP head",
    )
    parser.add_argument(
        "--fr-vocab-size",
        type=int,
        help="Number of ranked token IDs to retain (defaults to the complete list)",
    )
    args = parser.parse_args()

    build_model(
        args.base.resolve(),
        args.head_source.resolve(),
        args.output.resolve(),
        args.frequency_token_ids.resolve() if args.frequency_token_ids else None,
        args.fr_vocab_size,
        args.source_head_name,
        args.replace_existing_head,
    )


if __name__ == "__main__":
    main()
