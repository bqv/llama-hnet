# llama-hnet

An experimental fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that adds an `hnet` architecture for
[H-Nets](https://arxiv.org/abs/2507.07955) - byte-level language models with learned dynamic chunking.

This is **not** upstream and will not be proposed upstream: it is AI-assisted work whose author cannot claim the
maintenance commitment that llama.cpp requires. Treat it as a research fork that happens to borrow ggml.

## Status

| piece | state |
|---|---|
| GGUF schema (`hnet` arch, 28 tensor types, 13 metadata keys) | done |
| converter from safetensors | done, `convert_hnet_to_gguf.py` |
| model loading, tokeniser-less byte vocab | done, verified on a 195M Finnish checkpoint |
| inference graph | **not implemented** - `build_arch_graph` aborts by design |
| numeric parity against the PyTorch reference | next |

Loading works end to end; the fork prints parameters and then stops at:

```
hnet: graph is not implemented
```

## What an H-Net is

Three stacks and a router:

- an encoder of Mamba-2 layers over raw bytes,
- a router that scores every byte boundary by cosine distance between adjacent hidden states,
- a main transformer that runs only at the positions the router keeps (about one in four bytes),
- a decoder of Mamba-2 layers that expands chunk states back to bytes with an EMA.

Vocabulary is 256 bytes. There is no tokeniser, and the GGUF therefore carries no `tokenizer.*` keys.

## Build

```sh
cmake -B build -G Ninja -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 --target llama-cli llama-server
```

CPU is enough for the tests. ROCm/CUDA work as in upstream llama.cpp.

## Convert a checkpoint

```sh
python convert_hnet_to_gguf.py --input path/to/model-dir --output model.gguf
```

The input directory needs `config.json` and `model.safetensors` in the format written by
[bqv/hnets](https://github.com/bqv/hnets).

Two layout facts the converter enforces, because getting them wrong loads cleanly and produces nonsense:

- torch `Linear` weights are `[out, in]`; ggml wants `[in, out]`, so every 2-D weight is transposed, except
  `token_embd`;
- mamba `conv1d.weight` is `[d_inner, 1, d_conv]` in torch but `ggml_ssm_conv` requires the matrix
  `[d_conv, d_inner]`.

## Testing

The plan is numeric parity against the reference PyTorch stack, not a code review: same weights, same byte
sequence, compare logits element-wise at 1e-3, first on a 2.4M-parameter config, then on the 195M Finnish one.
If the logits agree, the C++ is right in the only sense that matters.

## Disclosure

The `hnet` architecture, converter and loading path in this fork were written with substantial AI assistance
(DeepSeek Harness). The design decisions are the owner's; the code has not been reviewed line by line by a human
and should be treated accordingly. Upstream llama.cpp explicitly forbids AI-generated contributions, which is one
of the reasons this lives here.

## Licence

MIT, inherited from llama.cpp. H-Net is by Sukjun Hwang, Brandon Wang and Albert Gu.
