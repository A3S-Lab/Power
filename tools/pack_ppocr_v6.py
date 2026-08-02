#!/usr/bin/env python3
"""Convert pinned PP-OCRv6 ONNX containers into native Power model assets.

ONNX is an offline interchange input only. The generated SafeTensors weights
and embedded A3S graph plans are the complete runtime inputs; a3s-power never
loads or executes ONNX.

Requires the development-only Python packages ``onnx``, ``numpy``, and
``safetensors``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file


SCHEMA_VERSION = 1
SUPPORTED_OPS = {
    "Add",
    "AveragePool",
    "BatchNormalization",
    "Concat",
    "Conv",
    "ConvTranspose",
    "Div",
    "Erf",
    "GlobalAveragePool",
    "HardSigmoid",
    "Identity",
    "MatMul",
    "MaxPool",
    "Mul",
    "Pow",
    "ReduceMean",
    "Relu",
    "Reshape",
    "Resize",
    "Shape",
    "Sigmoid",
    "Slice",
    "Softmax",
    "Sqrt",
    "Squeeze",
    "Sub",
    "Transpose",
    "Unsqueeze",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attribute_value(attribute: onnx.AttributeProto) -> Any:
    value = onnx.helper.get_attribute_value(attribute)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, list)):
        return value
    raise ValueError(
        f"unsupported attribute {attribute.name!r} of type "
        f"{onnx.AttributeProto.AttributeType.Name(attribute.type)}"
    )


def tensor_shape(value: onnx.ValueInfoProto) -> list[int | str]:
    result: list[int | str] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.dim_value:
            result.append(int(dimension.dim_value))
        elif dimension.dim_param:
            result.append(dimension.dim_param)
        else:
            result.append("dynamic")
    return result


def convert(source: Path, role: str, output: Path) -> None:
    model = onnx.load(str(source), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    unsupported = sorted({node.op_type for node in model.graph.node} - SUPPORTED_OPS)
    if unsupported:
        raise ValueError(f"unsupported PP-OCRv6 operators: {unsupported}")

    tensors: dict[str, np.ndarray[Any, Any]] = {}
    initializers: list[dict[str, Any]] = []
    for initializer in model.graph.initializer:
        value = np.ascontiguousarray(numpy_helper.to_array(initializer))
        if value.dtype == np.float64:
            value = value.astype(np.float32)
        if value.dtype not in (np.float32, np.float16, np.int64, np.int32):
            raise ValueError(
                f"initializer {initializer.name!r} has unsupported dtype {value.dtype}"
            )
        tensors[initializer.name] = value
        initializers.append(
            {
                "name": initializer.name,
                "dtype": str(value.dtype),
                "shape": list(value.shape) or [1],
            }
        )

    nodes = []
    for index, node in enumerate(model.graph.node):
        nodes.append(
            {
                "name": node.name or f"{node.op_type}.{index}",
                "op": node.op_type,
                "inputs": [name for name in node.input if name],
                "outputs": list(node.output),
                "attributes": {
                    attribute.name: attribute_value(attribute)
                    for attribute in node.attribute
                },
            }
        )

    plan = {
        "schemaVersion": SCHEMA_VERSION,
        "family": "pp-ocr-v6-small",
        "role": role,
        "source": {
            "format": "onnx",
            "sha256": sha256(source),
            "opset": max(entry.version for entry in model.opset_import),
        },
        "inputs": [
            {"name": value.name, "shape": tensor_shape(value)}
            for value in model.graph.input
            if value.name not in tensors
        ],
        "outputs": [
            {"name": value.name, "shape": tensor_shape(value)}
            for value in model.graph.output
        ],
        "initializers": sorted(initializers, key=lambda item: item["name"]),
        "nodes": nodes,
    }

    output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output / "model.safetensors"))
    (output / "graph.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--det", type=Path, required=True)
    parser.add_argument("--rec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    convert(arguments.det, "detection", arguments.output / "det")
    convert(arguments.rec, "recognition", arguments.output / "rec")


if __name__ == "__main__":
    main()
