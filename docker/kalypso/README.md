# Kalypso container on vLLM 0.29.0

This Dockerfile ports the Kalypso Python package to the official NVIDIA CUDA
vLLM `v0.29.0` image. The historical v0.13 fork in this repository remains
unchanged. The image uses the new release's compiled libraries and Python
internals, with narrow patches for semantic routes and KV ownership.

Upstream source reviewed: `98dff2a81d747d1dba01a47f939f48c3526d4206`.
The build checks the installed version and fails if a patch location changes.
Do not substitute a floating `latest` tag without porting and testing again.

## Build

From the repository root (`kalypso/`), build natively on the GH200 ARM64 host:

```bash
docker build --platform linux/arm64 -f docker/Dockerfile.kalypso -t kalypso:vllm-0.29.0 .
docker run --rm kalypso:vllm-0.29.0 --help
```

The build needs access to Docker Hub and enough disk for the upstream CUDA
image. Model weights are downloaded at runtime or supplied through a volume.
The base image must provide a `linux/arm64` manifest. An image built for
`linux/amd64` on the development host cannot be used natively on GH200.
See [vLLM's ARM64 container guidance](https://docs.vllm.ai/en/latest/deployment/docker/).

## Run

The requested deployment model is
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).
**Qwen compatibility is not complete:** its nested text configuration and
hybrid linear/full attention cache layout require changes to Kalypso's memory
estimator. The command below records the intended deployment configuration;
it is not yet a working Kalypso startup recipe for this model.

Target: two GH200 GPUs visible to the same container, with tensor parallelism
of 2 and a vLLM memory budget of 45% of each GPU's memory. This setting covers
weights, runtime allocations and cache; it is not a GPU compute-utilization
limit. Requires NVIDIA Container Toolkit configured on the host.

```bash
docker run --rm --name kalypso --gpus '"device=0,1"' --shm-size=16g \
  -p 8000:8000 \
  -e HF_TOKEN \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD/vllm/kalypso/benchmark/sample_data:/data:ro" \
  kalypso:vllm-0.29.0 Qwen/Qwen3.8-27B \
  --host 0.0.0.0 --port 8000 \
  --enable-prefix-caching --no-async-scheduling \
  --api-server-count 1 --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.45 --dtype bfloat16 \
  --max-model-len 32768
```

Set `HF_TOKEN` in the host environment if the model requires authentication.
Hybrid-cache budgeting and pin/release behavior must be validated for
Qwen before deployment. Speculative decoding and disaggregated KV transfer
are not validated.
Data parallelism and multiple API processes are rejected because Kalypso's
budget and pin registry are local to one API process. ICP and cascade services
remain separate services; their URLs must be reachable from this container.

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/v1/semantic/healthz
curl --fail http://localhost:8000/v1/semantic/pinned
```

Use existing benchmark clients with their API base pointing to port 8000 and
`data_path` referring to a mounted **container** path such as `/data/articles_500`.
After a query completes, check that the pinned registry is empty. Test both
semantic queries and ordinary `/v1/chat/completions` before deployment.
The built-in healthcheck assumes port 8000.

## Export a portable image archive

```bash
docker save kalypso:vllm-0.29.0 | gzip > kalypso-vllm-0.29.0.tar.gz
# On another Docker host:
docker load < kalypso-vllm-0.29.0.tar.gz
```

## Apptainer / SIF on GH200

Docker is not required. The definition file uses the same upstream image and
`prepare.py` patches as the Dockerfile. From `kalypso/`, on an ARM64 host:

```bash
ml load apptainer/latest
apptainer build --fakeroot kalypso-vllm-0.29.0.sif docker/kalypso.def
```

The definition uses the runtime image's `python3`, including for the vLLM
entrypoint. `/opt/venv/bin/python` belongs to an upstream build stage and is
not assumed to exist in the final image. If an older copy failed with
`python: not found`, rebuild with the updated definition above; changing
fakeroot options does not fix that missing executable.

The definition creates empty mount destinations inside the image in `%setup`,
before Apptainer enters `%post`, matching the cluster's `APPTAINER_BINDPATH`:
`/work,/nese,/project,/gypsum,/scratch,/modules,/nas,/datasets,/scratch4,/scratch3,/rstor`.
This does not create directories on the host or copy their contents into the
image. Creating these destinations in `%post` would be too late to fix the
mount failure.

The cluster must permit fakeroot builds, and the base tag must have an ARM64
image. Use a site-approved build host if compute nodes do not permit builds.
Do not build on x86-64 and assume the SIF will run natively on GH200.
No weights are embedded in the SIF.

Once Qwen compatibility is completed, the intended two-GPU launch is:

```bash
mkdir -p "$PWD/.container-cache"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
apptainer run --nv --cleanenv \
  --bind "$PWD/.container-cache:/cache" \
  --bind "$PWD/vllm/kalypso/benchmark/sample_data:/data:ro" \
  --pwd /data \
  kalypso-vllm-0.29.0.sif Qwen/Qwen3.8-27B \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.45 \
  --dtype bfloat16 --max-model-len 32768 \
  --enable-prefix-caching --no-async-scheduling --api-server-count 1
```

This assumes both GPUs are on one host/allocation. Preserve scheduler-provided
`CUDA_VISIBLE_DEVICES` as shown. Apptainer uses host networking by default, so
no Docker-style `-p` option is needed. The writable cache bind is needed because
the SIF filesystem is read-only. For gated models, explicitly forward a token
with `APPTAINERENV_HF_TOKEN`; `--cleanenv` removes ordinary host variables.

To convert an already-built ARM64 Docker image instead:

```bash
apptainer build kalypso-vllm-0.29.0.sif docker-daemon:kalypso:vllm-0.29.0
```

References: [OCI image builds](https://apptainer.org/docs/user/latest/docker_and_oci.html)
and [GPU support](https://apptainer.org/docs/user/main/gpu.html).
The definition has not been built here: this x86-64 session's Apptainer exits
with `Couldn't determine user account information: user: unknown userid 2732`.
Packaging in SIF does not resolve the unfinished Qwen memory-estimator port.

## Port validation

CPU ownership regressions (no vLLM installation required):

```bash
uv run --no-project python docker/kalypso/test_engine.py
```

To validate patch locations against an unpacked upstream v0.29.0 source tree:

```bash
uv run --no-project python docker/kalypso/prepare.py \
  --source "$PWD" --target /path/to/vllm-0.29.0/vllm
uv run --no-project python docker/kalypso/test_scheduler.py /path/to/vllm-0.29.0
```

This modifies the target source tree. Run it on a disposable checkout.
CPU checks do not establish GPU inference compatibility; the image build,
startup and end-to-end semantic query tests must also pass on a GPU host.

Validation performed during this port: patch application against the upstream
commit above, Python compilation, three ownership tests and three tests of the
patched upstream scheduler finish method. No working container builder was available
in the development session, so the image has not been built and GPU inference
has not been tested. Treat this as a port candidate until those checks pass.
