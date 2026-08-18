#!/usr/bin/env python3
"""Add a separately quantized MTP output head to a GGUF model.

The base model is copied without changing any existing tensor. The tensor named
``output.weight`` is read from a second GGUF and appended under the Qwen NextN
head name, allowing the target context to keep its original output projection
while an MTP draft context uses the cheaper projection.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

try:
    import gguf
except ImportError as error:
    raise SystemExit(
        "The llama.cpp gguf Python package is required; add gguf-py to PYTHONPATH"
    ) from error


SOURCE_HEAD_NAME = "output.weight"
MTP_HEAD_NAME = "blk.64.nextn.shared_head_head.weight"


def field_value(reader: gguf.GGUFReader, key: str) -> Any:
    field = reader.get_field(key)
    return field.contents() if field else None


def copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
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


def build_model(base_path: Path, head_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    partial_path = output_path.with_name(f"{output_path.name}.partial")
    if partial_path.exists():
        raise FileExistsError(f"refusing to overwrite {partial_path}")

    base = gguf.GGUFReader(base_path, "r")
    head_source = gguf.GGUFReader(head_path, "r")

    if any(tensor.name == MTP_HEAD_NAME for tensor in base.tensors):
        raise ValueError(f"base GGUF already contains {MTP_HEAD_NAME!r}")

    base_head = tensor_by_name(base, SOURCE_HEAD_NAME)
    draft_head = tensor_by_name(head_source, SOURCE_HEAD_NAME)
    if tuple(base_head.shape) != tuple(draft_head.shape):
        raise ValueError(
            f"head shape mismatch: base={tuple(base_head.shape)}, "
            f"draft={tuple(draft_head.shape)}"
        )
    if base_head.tensor_type == draft_head.tensor_type:
        raise ValueError(
            "draft head has the same quantization type as the base output tensor"
        )

    architecture = field_value(base, gguf.Keys.General.ARCHITECTURE)
    if not isinstance(architecture, str):
        raise ValueError("base GGUF has no valid general.architecture")

    writer = gguf.GGUFWriter(partial_path, arch=architecture, endianess=base.endianess)
    alignment = field_value(base, gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment

    copy_metadata(base, writer)
    for tensor in base.tensors:
        add_tensor_info(writer, tensor, tensor.name)
    add_tensor_info(writer, draft_head, MTP_HEAD_NAME)

    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()

        total = len(base.tensors) + 1
        for index, tensor in enumerate(base.tensors, start=1):
            writer.write_tensor_data(tensor.data, tensor_endianess=base.endianess)
            if index % 100 == 0:
                print(f"Copied {index}/{total} tensors", flush=True)

        writer.write_tensor_data(draft_head.data, tensor_endianess=head_source.endianess)
        writer.close()
        os.replace(partial_path, output_path)
    except BaseException:
        writer.close()
        partial_path.unlink(missing_ok=True)
        raise

    print(
        f"Added {MTP_HEAD_NAME} as {draft_head.tensor_type.name}; "
        f"wrote {output_path}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="Original target GGUF")
    parser.add_argument("head_source", type=Path, help="GGUF containing the quantized output.weight")
    parser.add_argument("output", type=Path, help="Output GGUF with a separate MTP head")
    args = parser.parse_args()

    build_model(args.base.resolve(), args.head_source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
