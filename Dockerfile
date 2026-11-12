# syntax=docker/dockerfile:1
# LoRA training worker for the SaladCloud / Kelpie pilot.
# Contains only generic code (PyTorch + kohya sd-scripts + Kelpie + entrypoint).
# No model, no dataset, no credentials are baked into this image.
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates unzip \
    && rm -rf /var/lib/apt/lists/*

# kohya-ss sd-scripts, pinned to the exact commit used on the Mac training setup
ARG SD_SCRIPTS_REF=cb70b96858477e716a2096292ab8aab433277f38
RUN git clone https://github.com/kohya-ss/sd-scripts /app/sd-scripts \
    && cd /app/sd-scripts && git checkout ${SD_SCRIPTS_REF}

WORKDIR /app/sd-scripts
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Kelpie worker (version pinned; matches the official Salad Kelpie guide)
ARG KELPIE_VERSION=0.7.2
ADD https://github.com/SaladTechnologies/kelpie/releases/download/${KELPIE_VERSION}/kelpie /kelpie
RUN chmod +x /kelpie

COPY train_entry.sh /app/train_entry.sh
RUN chmod +x /app/train_entry.sh

RUN mkdir -p /work/input/models /work/input/dataset /work/output

# Kelpie is the main process; job definitions supply the command it runs.
CMD ["/kelpie"]
