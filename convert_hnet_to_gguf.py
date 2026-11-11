import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent / "gguf-py"))

from gguf import GGUFWriter, GGMLQuantizationType

STACKS = [
    ("backbone.encoder.layers.", "enc."),
    ("backbone.main_network.main_network.layers.", "blk."),
    ("backbone.decoder.layers.", "dec."),
]

SUFFIXES = [
    "mixer.in_proj.weight",
    "mixer.conv1d.weight",
    "mixer.conv1d.bias",
    "mixer.dt_bias",
    "mixer.A_log",
    "mixer.D",
    "mixer.norm.weight",
    "mixer.out_proj.weight",
    "mixer.Wqkv.weight",
    "mixer.out_qkv.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
    "norm1.weight",
    "norm2.weight",
]

SINGLE = {
    "embeddings.weight": "token_embd.weight",
    "lm_head.weight": "output.weight",
    "backbone.encoder.rmsnorm.weight": "enc.rmsnorm.weight",
    "backbone.main_network.main_network.rmsnorm.weight": "blk.rmsnorm.weight",
    "backbone.decoder.rmsnorm.weight": "dec.rmsnorm.weight",
    "backbone.routing_module.q_proj_layer.weight": "router.q_proj_layer.weight",
    "backbone.routing_module.k_proj_layer.weight": "router.k_proj_layer.weight",
    "backbone.residual_proj.weight": "residual_proj.weight",
    "backbone.residual_proj.bias": "residual_proj.bias",
}


def map_name(name):
    if name in SINGLE:
        return SINGLE[name]
    for prefix, out in STACKS:
        if name.startswith(prefix):
            rest = name[len(prefix):]
            index, _, tail = rest.partition(".")
            if tail in SUFFIXES:
                return f"{out}{index}.{tail}"
    return None


def count_layers(names, prefix):
    found = {int(n[len(prefix):].split(".")[0]) for n in names if n.startswith(prefix)}
    return max(found) + 1 if found else 0


def read_config(path):
    cfg = json.load(open(path))
    d_model = cfg["d_model"]
    layout = cfg["arch_layout"]
    dec_stage = layout[0]
    main_stage = layout[1][0]
    enc_stage = layout[2]
    n_enc = int(dec_stage[1:])
    n_main = int(main_stage[1:])
    n_dec = int(enc_stage[1:])
    return {
        "d_enc": int(d_model[0]),
        "d_main": int(d_model[1]),
        "d_dec": int(d_model[0]),
        "n_enc": n_enc,
        "n_dec": n_dec,
        "n_main": n_main,
        "d_ff_main": int(cfg["d_intermediate"][1]),
        "n_head_main": int(cfg["attn_cfg"]["num_heads"][1]),
        "rotary_emb_dim": int(cfg["attn_cfg"].get("rotary_emb_dim", [0, 0])[1] or 0),
        "n_head_enc": int(cfg["attn_cfg"]["num_heads"][0]),
        "head_dim_enc": int(d_model[0]) // int(cfg["attn_cfg"]["num_heads"][0]),
    }


def main():
    ap = argparse.ArgumentParser(description="Convert a Finnish H-Net safetensors directory to GGUF")
    ap.add_argument("--input", required=True, help="directory with config.json and model.safetensors")
    ap.add_argument("--output", required=True, help="output .gguf path")
    ap.add_argument("--target-ratio", type=float, default=4.0)
    ap.add_argument("--rope-base", type=float, default=10000.0)
    args = ap.parse_args()

    src = Path(args.input)
    hparams = read_config(src / "config.json")

    writer = GGUFWriter(args.output, "hnet")
    writer.add_name(src.name)
    writer.add_uint32("hnet.context_length", 4096)
    writer.add_uint32("hnet.embedding_length", hparams["d_main"])
    writer.add_uint32("hnet.d_enc", hparams["d_enc"])
    writer.add_uint32("hnet.d_main", hparams["d_main"])
    writer.add_uint32("hnet.d_dec", hparams["d_dec"])
    writer.add_uint32("hnet.n_enc", hparams["n_enc"])
    writer.add_uint32("hnet.n_dec", hparams["n_dec"])
    writer.add_uint32("hnet.n_main", hparams["n_main"])
    writer.add_uint32("hnet.n_head_main", hparams["n_head_main"])
    writer.add_uint32("hnet.n_head_enc", hparams["n_head_enc"])
    if hparams["rotary_emb_dim"] > 0:
        writer.add_uint32("hnet.rotary_emb_dim", hparams["rotary_emb_dim"])
    writer.add_uint32("hnet.head_dim_enc", hparams["head_dim_enc"])
    writer.add_uint32("hnet.d_ff_main", hparams["d_ff_main"])
    writer.add_float32("hnet.target_ratio", args.target_ratio)
    writer.add_float32("hnet.rope_base", args.rope_base)
    writer.add_string("hnet.dechunk_mode", "ema")

    written = 0
    skipped = []
    with safe_open(src / "model.safetensors", "pt") as f:
        for name in sorted(f.keys()):
            target = map_name(name)
            if target is None:
                skipped.append(name)
                continue
            tensor = f.get_tensor(name)
            if tensor.ndim == 3 and "conv1d.weight" in target:
                tensor = tensor.squeeze(1).contiguous()
            elif tensor.ndim == 2:
                tensor = tensor.contiguous()
            if tensor.ndim <= 1 or target.endswith("norm.weight") or target.endswith("rmsnorm.weight") or target.endswith("conv1d.weight"):
                writer.add_tensor(target, tensor.to(torch.float32).numpy())
            elif tensor.dtype == torch.bfloat16:
                array = tensor.view(torch.int16).numpy()
                writer.add_tensor(target, array, raw_dtype=GGMLQuantizationType.BF16)
            else:
                writer.add_tensor(target, tensor.to(torch.float16).numpy(),
                                  raw_dtype=GGMLQuantizationType.F16)
            written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"wrote {args.output}: {written} tensors")
    print(f"hparams {hparams}")
    for name in skipped:
        print(f"skipped {name}")


if __name__ == "__main__":
    main()
