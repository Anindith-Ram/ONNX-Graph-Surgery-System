#!/usr/bin/env python3
"""
Print an ONNX graph as:
<node id>, <op>, <input node ids>  <input shape>, <output shape>

- Ignores initializers (weights/biases) as dependencies
- Ignores node attributes entirely
- Uses ONNX shape inference to get shapes (when possible)
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple, Union

import onnx
from onnx import shape_inference


Shape = Tuple[Union[int, str, None], ...]  # dims can be int, symbolic str, or None


def _shape_from_value_info(vi: onnx.ValueInfoProto) -> Optional[Shape]:
    """Extract (possibly partial) tensor shape from a ValueInfoProto."""
    tt = vi.type.tensor_type
    if tt is None or tt.shape is None:
        return None
    dims: List[Union[int, str, None]] = []
    for d in tt.shape.dim:
        # dim_value: concrete int, dim_param: symbolic string
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        elif d.HasField("dim_param"):
            dims.append(str(d.dim_param))
        else:
            dims.append(None)
    return tuple(dims)


def _collect_shape_table(g: onnx.GraphProto) -> Dict[str, Shape]:
    """
    Build a mapping: tensor_name -> shape tuple
    Sources: graph.input, graph.output, graph.value_info
    """
    table: Dict[str, Shape] = {}

    def add_vi(vi: onnx.ValueInfoProto):
        if not vi.name:
            return
        shp = _shape_from_value_info(vi)
        if shp is not None:
            table[vi.name] = shp

    for vi in g.input:
        add_vi(vi)
    for vi in g.output:
        add_vi(vi)
    for vi in g.value_info:
        add_vi(vi)

    return table


def _fmt_shape(shape: Optional[Shape]) -> str:
    if shape is None:
        return "?"
    # print like (1, 3, 224, 224) with ? for unknown dims
    parts = []
    for d in shape:
        if d is None:
            parts.append("?")
        else:
            parts.append(str(d))
    return "(" + ", ".join(parts) + ")"


def print_onnx_graph(path: str) -> None:
    # Load
    model = onnx.load(path)

    # Infer shapes (best-effort)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        # Shape inference can fail on some models; still print structure.
        print(f"[warn] shape_inference failed: {type(e).__name__}: {e}")
        print("[warn] continuing without inferred intermediate shapes...")

    g = model.graph

    # Initializers (weights/biases): treat as constants, NOT as dependency nodes
    initializer_names = {init.name for init in g.initializer}

    # Assign ids to real nodes (0..N-1)
    nodes = list(g.node)
    node_id: Dict[int, int] = {i: i for i in range(len(nodes))}  # index->id (same)

    # Shapes table: tensor_name -> shape
    shape_table = _collect_shape_table(g)

    # Producer map: tensor_name -> producing "node id" (or pseudo input id)
    producer: Dict[str, str] = {}

    # Create pseudo node ids for graph inputs (excluding initializers)
    # so dependencies can reference them.
    graph_input_names = [vi.name for vi in g.input if vi.name and vi.name not in initializer_names]
    for name in graph_input_names:
        producer[name] = f"in:{name}"

    # Map each node output tensor to the node id
    for i, n in enumerate(nodes):
        nid = str(node_id[i])
        for out_name in n.output:
            if out_name:  # can be empty in some models
                producer[out_name] = nid

    # Print nodes
    for i, n in enumerate(nodes):
        nid = str(node_id[i])
        op = n.op_type

        # For dependencies: only consider non-empty inputs that are not initializers
        data_inputs = [x for x in n.input if x and x not in initializer_names]

        input_node_ids = []
        input_shapes = []
        for x in data_inputs:
            input_node_ids.append(producer.get(x, f"unknown:{x}"))
            input_shapes.append(_fmt_shape(shape_table.get(x)))

        output_shapes = []
        for y in n.output:
            if not y:
                continue
            output_shapes.append(_fmt_shape(shape_table.get(y)))

        # Format exactly as requested
        # <node id>, <op>, <input node ids>  <input shape>, <output shape>
        # If multiple inputs/outputs exist, we print them as lists.
        print(
            f"{nid}, {op}, {input_node_ids}  "
            f"{input_shapes}, {output_shapes}"
        )

    # (Optional) also show graph outputs as "nodes"
    # so it's easy to see which node produces each model output.
    if g.output:
        print("\n# graph outputs")
        for out_vi in g.output:
            name = out_vi.name
            if not name:
                continue
            prod = producer.get(name, f"unknown:{name}")
            shp = _fmt_shape(shape_table.get(name))
            print(f"out:{name}, Identity, [{prod}]  [{shp}], [{shp}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_path", help="Path to ONNX model (e.g., super_resolution.onnx)")
    args = ap.parse_args()
    print_onnx_graph(args.onnx_path)


if __name__ == "__main__":
    main()
