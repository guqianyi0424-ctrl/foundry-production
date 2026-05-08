FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && conda init bash

RUN conda create -n foundry python=3.12 -y

SHELL ["conda", "run", "-n", "foundry", "/bin/bash", "-c"]

RUN pip install -e "git+https://github.com/RosettaCommons/foundry-production.git#egg=rc-foundry[all]" \
    || pip install rc-foundry[all]

RUN foundry install rfd3 ligandmpnn rf3 --checkpoint-dir /opt/checkpoints || true

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN conda run -n foundry pip install --no-cache-dir -r /app/backend/requirements.txt

COPY utils /app/utils
COPY config /app/config
COPY backend /app/backend

RUN mkdir -p /app/outputs /app/data

EXPOSE 8000

ENV PYTHONPATH=/app/backend:/app
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["conda", "run", "--no-banner", "-n", "foundry", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
