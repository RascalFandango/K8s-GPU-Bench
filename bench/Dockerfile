# syntax=docker/dockerfile:1
#
# Benchmark image for nrp-gpu-bench.
#
# Base: NVIDIA's PyTorch container -- torch + CUDA + cuDNN prebuilt and tuned,
# so we don't compile anything. The 24.12 tag publishes BOTH linux/amd64 and
# linux/arm64, which is load-bearing: the Grace/GH200 nodes are arm64 and will
# not run an amd64 image. Build multi-arch and push:
#
#   docker buildx build --platform linux/amd64,linux/arm64 \
#     -t <registry>/nrp-gpu-bench:latest --push bench/
#
# (Grace nodes also require tolerating nautilus.io/hardware=arm64; submit.py
# emits that toleration for arm64 flavors -- see POLICY.md.)
FROM nvcr.io/nvidia/pytorch:24.12-py3

# nvidia-ml-py provides the `pynvml` module (live NVML telemetry);
# prometheus_client provides the :9400 exporter and the Pushgateway push. torch
# and CUDA already live in the base image. Versions pinned for reproducibility
# -- bump deliberately, don't float.
RUN pip install --no-cache-dir \
      nvidia-ml-py==12.560.30 \
      prometheus_client==0.21.1

# Stream stdout/stderr unbuffered so `kubectl logs -f` shows progress live
# instead of appearing hung behind Python's line buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY bench.py /app/bench.py

# Exec form: the script is PID 1 and receives SIGTERM directly, so the Job
# terminates promptly. The script exits on its own -- no sleep infinity (§2.4).
ENTRYPOINT ["python", "/app/bench.py"]
