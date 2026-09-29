FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git tmux rsync ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace
COPY requirements.lock requirements.txt ./
RUN uv venv --python 3.12 .venv && \
    uv pip install --python .venv/bin/python -r requirements.lock

COPY . .
SHELL ["/bin/bash", "-lc"]
CMD ["bash"]
