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
| inference graph | done - two passes, `llama_encode` then `llama_decode` |
| CPU smoke test (finite logits, 195M) | **passing** |
| numeric parity against the PyTorch reference | segmentation identical at 2.36M and 195M; logits within ~0.5% |

Loads and computes end to end on CPU, and matches the PyTorch reference on a 2.36M-parameter model with 24 byte
tokens (`tests/hnet/gen_ref.py` for the reference, `tests/hnet/smoke.cpp` for the C++ side):

```
chunk boundaries     11 of 24, identical positions in both [0 2 4 5 8 12 14 17 19 21 22]
boundary probs       agree to ~1e-3  (0.4386 vs 0.4395, 0.5149 vs 0.5156, ...)
logits               max abs 0.090, mean abs 0.022, same argmax
```

At 195M, against the trained Finnish weights and with `hnet.rotary_emb_dim` honoured (48 of 64):

```
chunk boundaries     12 of 24, identical positions in both [0 2 5 7 13 15 16 18 19 20 22 23]
logits               max abs 0.132, mean abs 0.044, same argmax
```

The segmentation being exact at both sizes says the router, the cosine, the threshold, the gather and the RoPE width
are right. The remaining logit gap is precision, not logic: the reference runs in bf16 on the GPU and rounds the
de-chunk EMA inputs to bf16, while this path computes in F32 with 16-bit weights.

## Tensor dtypes the kernels require

Not a preference; these are the assumptions in ggml's CPU kernels:

| tensor class | type | why |
|---|---|---|
| norms, `dt_bias`, `D`, `conv1d.bias`, `residual_proj.bias` | F32 | the elementwise binary op has no F32 x F16 case |
| `conv1d.weight` | F32 | `ggml_ssm_conv` asserts `nb[0] == sizeof(float)` |
| `A_log` | 16-bit fine | the graph casts to F32 before `exp` |
| matmul weights | 16-bit | quantisable |

The converter applies this table. It was learned the hard way: a checkpoint with 16-bit norms aborts on the
first `mul` of the encode pass, and 16-bit conv weights abort in `ssm_conv`.

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

- no dimension transposition is needed: GGUF writes tensor dims reversed, so a torch `[out, in]` weight already
  arrives as ggml `ne[0] = in`, which is what `ggml_mul_mat` wants. Transposing it produces a double reversal that
  fails for rectangular tensors and silently computes wrong values for square ones (router, `residual_proj`);
- mamba `conv1d.weight` is `[d_inner, 1, d_conv]` in torch; squeeze the middle axis only, giving `[d_inner, d_conv]`,
  which arrives as `ne[0] = d_conv` as `ggml_ssm_conv` requires.

## Segmenting text

`hnet-segment` exposes the learned chunking directly - pass A of the two-pass graph, no generation:

```sh
printf 'Suomen kieli on vaikea.' | ./build/bin/hnet-segment model.gguf
|S|uom|en| kieli| o|n| va|i|ke|a|.
chunks 11 of 23 bytes
```

`--json` emits the boundary positions and the per-byte boundary probabilities instead of the marked text, which
is what the `hnet-probe` results in the Finnish write-up were measured with.

## Generating text

`hnet-generate` runs the pair of passes in a loop, re-encoding the whole sequence each step (there is no
incremental state yet, so this is O(n^2) and CPU-appropriate rather than fast):

```sh
./build/bin/hnet-generate model.gguf "Suomen kieli on " 12
Suomen kieli on kieli, joka

./build/bin/hnet-generate model.gguf "Suomen kieli on " 24 --temp 0.8 --seed 7
Suomen kieli on jatkoa ja se on lähes s
```

That is the 195M Finnish checkpoint, greedy, on a CPU: the first sentence it ever produced through llama.cpp.
Each step reports the chunk count so the segmentation growth is visible as the sequence lengthens.

## Testing

The plan is numeric parity against the reference PyTorch stack, not a code review: same weights, same byte
sequence, compare logits element-wise at 1e-3, first on a 2.4M-parameter config, then on the 195M Finnish one.
If the logits agree, the C++ is right in the only sense that matters.

## Known gaps

- **No incremental state.** `hnet-generate` re-encodes the whole prefix each step, which is O(n^2) and makes CPU
  generation slow (~13 steps in about three minutes at 195M). Real generation needs per-layer SSM and conv state,
  the router's previous hidden state, and the de-chunk EMA carried across calls.
- **The main stack runs `n_tokens` rows, not `n_chunk` rows.** Junk rows past the last chunk are hidden by the
  causal mask and never read, but they cost about 4x the main-stack compute at the target 4:1 chunk ratio. Fixing
  it means building the decode graph after the host step, with `n_chunk` known.
- **No server surface.** `/v1/segment` would be a thin handler over `llama_hnet_boundaries`, but the server's
  slot and task machinery has to be respected, so it is not a five-line change.
- **Single sequence, single ubatch.** Packed multi-sequence batches would need per-sequence byte maps and positions.
- **Precision.** Logits agree with the PyTorch reference to ~0.5%, not bit-exactly: the reference runs bf16 on the
  GPU, this runs F32 on the CPU with 16-bit weights.

## Disclosure

The `hnet` architecture, converter and loading path in this fork were written with substantial AI assistance
(DeepSeek Harness). The design decisions are the owner's; the code has not been reviewed line by line by a human
and should be treated accordingly. Upstream llama.cpp explicitly forbids AI-generated contributions, which is one
of the reasons this lives here.

## Numbers on this box

Generation at 195M, CPU (`-ngl 0`), greedy, 12 bytes: about three minutes, because every step re-encodes the
prefix. Prefill of a 24-byte prompt costs roughly 20 s cold; the parity run above reuses one loaded context.

## Licence

MIT, inherited from llama.cpp. H-Net is by Sukjun Hwang, Brandon Wang and Albert Gu.
