# Run Training Factory on your own cloud

`thousandml` is a small CLI that trains and serves CQ models with GCMP or LoRA adapters on **your** cloud account: AWS, GCP, Azure, RunPod, Lambda, Kubernetes and 15+ others. It runs on [SkyPilot](https://github.com/skypilot-org/skypilot). Your data, adapters and GPUs stay in your account; ThousandML never sees them.

## Setup (5 minutes)

```bash
pip install "thousandml[aws]"          # or [gcp], [azure], [runpod], [kubernetes]
sky check                              # confirms your cloud credentials work

export CQ_LICENSE_KEY=...              # from contact@thousandml.com
export HF_TOKEN=...                    # only for private Hugging Face models
```

From a repo checkout you can also use `pip install "./cli[aws]"`.

## Train an adapter

```bash
thousandml train support-v1 \
  --data s3://my-bucket/support/train.jsonl \
  --output s3://my-bucket/adapters \
  --adapter gcmp --max-steps 500
```

- **Data:** each row is `{"text": ...}` or `{"messages": [...]}`. A local file path works too.
- **Base precision:** see the next section. By default you get CQ 4-bit if it exists for your model, and NF4 otherwise.
- **GPU:** SkyPilot picks the cheapest available one from `A100-80GB, H100, L40S, A40`. Use `--gpus H100:1` to choose, and `--spot` for spot pricing.
- **Output:** the adapter lands in `s3://my-bucket/adapters/support-v1/`.
- **Teardown:** the cluster shuts down when training finishes. Add `--keep` to leave it running.
- **LoRA:** `--adapter lora --rank 16` trains a LoRA baseline instead.

## Choose the base: CQ, NF4 or full precision

Pass the model's normal Hugging Face id with `--model`. Then choose what the adapter trains on top of with `--precision`:

| `--precision` | Base the adapter trains on | Default GPUs |
|---|---|---|
| `auto` (default) | CQ if `thousandml/<model>-CQ-4bit` exists, otherwise NF4 | 48 GB or more |
| `cq` | ThousandML CQ 4-bit; stops with an error if there is no CQ checkpoint for the model | 48 GB or more |
| `nf4` | bitsandbytes NF4 4-bit: Unsloth's pre-quantized checkpoint if one is published, otherwise quantized when loaded | 48 GB or more |
| `bf16` | Full precision, no quantization | 80 GB or more |

```bash
thousandml train support-v1 --model Qwen/Qwen3.8-27B --data ... --output ...                     # CQ (auto)
thousandml train support-v1 --model Qwen/Qwen3.8-27B --precision bf16 --data ... --output ...    # full precision
thousandml train support-v1 --model mistralai/Mistral-7B-v0.3 --data ... --output ...          # no CQ yet -> NF4
```

The CLI prints the base it chose before it starts a machine. GCMP and LoRA work on all three bases. Each adapter folder gets a `thousandml.json` that records its base. The GPU defaults above assume a model the size of Qwen3.8-27B; pass `--gpus` for other sizes.

## Serve every adapter behind one endpoint

```bash
thousandml serve prod --adapters s3://my-bucket/adapters --replicas 1-4
```

- **Adapters:** every sub-folder of `--adapters` that has `adapter_config.json` is loaded at startup. All of them share one copy of the base model.
- **Endpoint:** an OpenAI-compatible API. Choose the adapter per request with `"model": "<adapter-name>"`; `"model": "base"` uses no adapter.
- **Autoscaling:** `--replicas 1-4` adds replicas as traffic grows. A fixed number such as `--replicas 1` turns it off.
- **Base:** pass the same `--model` and `--precision` you trained with. Adapters trained on a different base are skipped, with a message in the logs.
- **What vLLM can serve:**

  | Base | GCMP adapters | LoRA adapters |
  |---|---|---|
  | CQ | yes | yes |
  | NF4 / BF16 | not yet | yes |

  Put GCMP and LoRA adapters in separate folders and serve them as two services.

```bash
sky serve status prod                  # shows the endpoint URL
curl http://<endpoint>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"support-v1","messages":[{"role":"user","content":"Hi"}]}'
```

## Other commands

```bash
thousandml status            # clusters and services
thousandml logs support-v1   # follow training logs
thousandml down prod         # stop a service or cluster
thousandml train ... --dry-run   # print the SkyPilot YAML without launching
```

## Common options

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `Qwen/Qwen3.8-27B` | The model's normal Hugging Face id (a CQ or NF4 checkpoint id also works) |
| `--precision` | `auto` | `cq`, `nf4` or `bf16` (see above) |
| `--gpus` | depends on precision | Any one of these, e.g. `A100-80GB:1,A40:1` |
| `--cloud` | cheapest | `aws`, `gcp`, `azure`, `runpod`, ... |
| `--spot` | off | Use spot or preemptible GPUs |
| `--disk` | 200 GB | Boot disk size |
| `--image` | `thousandml/training-factory:latest` | Container image |

---

## Maintainers: build and push the image

The CLI runs everything inside one image that holds training, CQ kernels, GCMP and vLLM. Build it from the repo root after pulling the LFS binaries:

```bash
git lfs pull
docker build -f docker/Dockerfile.full -t thousandml/training-factory:0.1.0 .
docker tag thousandml/training-factory:0.1.0 thousandml/training-factory:latest
docker push thousandml/training-factory:0.1.0
docker push thousandml/training-factory:latest
```

The build fails if vLLM can't find the CQ quantization method or the GCMP multi-adapter op.

## Maintainers: publish the CLI

```bash
cd cli && python -m build && twine upload dist/*
```
