import sys, json, torch
from pathlib import Path
from safetensors.torch import load_file, save_file
from safetensors import safe_open
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
f = src / "adapter_model.safetensors"
meta = safe_open(str(f), "pt").metadata()
t = {k: v.to(torch.bfloat16).contiguous() for k, v in load_file(str(f)).items()}
save_file(t, str(dst / "adapter_model.safetensors"), metadata=meta)
(dst / "adapter_config.json").write_text((src / "adapter_config.json").read_text())
print("saved", dst, sum(v.numel() * 2 for v in t.values()) / 1e6, "MB")