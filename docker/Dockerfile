ARG CUDA_IMAGE=nvidia/cuda:12.8.1-base-ubuntu24.04
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.22
FROM ${UV_IMAGE} AS uv-bin
FROM ${CUDA_IMAGE} AS runtime-base
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy \
    PATH=/opt/runtime/bin:${PATH} \
    PYTHONPATH=/workspace:/workspace/external/opendde:/workspace/external/plip \
    PLIP_COMMAND=/usr/local/bin/plipcmd.py

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev gcc libc6-dev ca-certificates \
    hmmer kalign libglib2.0-0 libgl1 libxrender1 libgomp1 libatomic1 \
    openbabel python3-openbabel python3-lxml python3-numpy \
    && rm -rf /var/lib/apt/lists/*

FROM runtime-base AS build-base
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv build-essential cmake git curl pkg-config zlib1g-dev libbz2-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv-bin /uv /usr/local/bin/uv

FROM build-base AS foldmason-build
RUN apt-get update && apt-get install -y --no-install-recommends cargo rustc \
    && rm -rf /var/lib/apt/lists/*
ARG FOLDMASON_REV
RUN [[ "$FOLDMASON_REV" =~ ^[0-9a-f]{40}$ ]] \
    && mkdir -p /opt/foldmason \
    && curl --retry 3 --connect-timeout 30 --max-time 1800 -fsSL "https://api.github.com/repos/steineggerlab/foldmason/tarball/$FOLDMASON_REV" | tar xz --strip-components=1 -C /opt/foldmason \
    && cmake -S /opt/foldmason -B /opt/foldmason/build -DCMAKE_BUILD_TYPE=Release -DHAVE_AVX2=1 -DVERSION_OVERRIDE="$FOLDMASON_REV" \
    && cmake --build /opt/foldmason/build --parallel 2

FROM build-base AS compute-build
RUN python3 -m venv /opt/runtime
COPY docker/environment.json /tmp/environment.json

ARG PYPI_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
ARG PYTORCH_WHEEL_BASE=https://mirrors.nju.edu.cn/pytorch/whl
ARG TRITON_WHEEL_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/24/5f/950fb373bf9c01ad4eb5a8cd5eaf32cdf9e238c02f9293557a2129b9c4ac/triton-3.3.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
RUN --mount=type=cache,target=/root/.cache/uv \
    python3 -c 'import json; from pathlib import Path; p = json.loads(Path("/tmp/environment.json").read_text())["packages"]; Path("/tmp/environment-requirements.txt").write_text("\n".join(f"{name}=={version}" for name, version in p.items()) + "\n")' \
    && \
    UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 UV_CONCURRENT_DOWNLOADS=4 \
    uv pip install --python /opt/runtime/bin/python --index-url "$PYPI_INDEX_URL" \
    "torch @ $PYTORCH_WHEEL_BASE/cu128/torch-2.7.1%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=0b64f7d0a6f2a739ed052ba959f7b67c677028c9566ce51997f9f90fe573ddaa" \
    "triton @ $TRITON_WHEEL_URL#sha256=9999e83aba21e1a78c1f36f21bce621b77bcaa530277a50484a7cb4a822f6e43" \
    --requirement /tmp/environment-requirements.txt \
    && uv pip check --python /opt/runtime/bin/python \
    && uv pip freeze --python /opt/runtime/bin/python > /opt/runtime-packages.txt

FROM runtime-base AS release
WORKDIR /workspace
COPY --from=compute-build /opt/runtime /opt/runtime
COPY --from=compute-build /opt/runtime-packages.txt /opt/runtime-packages.txt
COPY --from=foldmason-build /opt/foldmason/build/src/foldmason /usr/local/bin/foldmason
COPY --from=foldmason-build /opt/foldmason/LICENSE.md /opt/licenses/FoldMason-LICENSE.md
COPY docker/environment.json /opt/compute-environment.json
RUN printf '%s\n' '#!/bin/sh' \
    'PYTHONPATH=/workspace/external/plip exec /usr/bin/python3 /workspace/external/plip/plip/plipcmd.py "$@"' \
    > /usr/local/bin/plipcmd.py && chmod +x /usr/local/bin/plipcmd.py \
    && dpkg-query -W > /opt/system-packages.txt
ARG ENVIRONMENT_ID
ARG ENVIRONMENT_SHA256
LABEL org.opendde-harness.environment=${ENVIRONMENT_ID} \
    org.opendde-harness.environment-sha256=${ENVIRONMENT_SHA256}
RUN --mount=type=bind,source=docker/doctor.py,target=/tmp/doctor.py \
    python /tmp/doctor.py --dependencies-only --environment /opt/compute-environment.json
EXPOSE 8080
CMD ["python"]
