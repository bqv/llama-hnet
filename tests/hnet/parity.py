import subprocess
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).parent
smoke = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "../../build/bin/hnet-smoke"
out = "/tmp/hnet-parity-logits.bin"

subprocess.run([str(smoke), str(here / "fixtures/tiny.gguf"), str(here / "fixtures/ids.txt"), out], check=True)

ref = np.load(here / "fixtures/ref.npz")
ref_last = ref["logits"][-1].astype(np.float64)
got = np.fromfile(out, dtype=np.float32).astype(np.float64)

diff = np.abs(ref_last - got)
print(f"logits max_abs {diff.max():.4f} mean_abs {diff.mean():.4f}")
print(f"argmax ref {int(ref_last.argmax())} got {int(got.argmax())}")
print(f"boundaries {int(ref['mask_0'].sum())}")

assert got.shape == ref_last.shape, (got.shape, ref_last.shape)
assert ref_last.argmax() == got.argmax(), "argmax differs"
assert diff.mean() < 0.05, f"mean abs diff too large: {diff.mean():.4f}"
print("parity ok")
