ARG BASE=nvidia/cuda:12.8.1-devel-ubuntu22.04
FROM ${BASE} AS wilor

# --- Corporate proxy root certificate fix ---
COPY docker/rootCA.cer /usr/local/share/ca-certificates/corporate-root-ca.crt
RUN chmod 644 /usr/local/share/ca-certificates/corporate-root-ca.crt && update-ca-certificates

# Ensure all SSL tools use the updated CA bundle
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# --------------------------------------------------------------------
# System Python + venv (no Anaconda)
# --------------------------------------------------------------------

# Install OS dependencies (same pattern as HAMER)
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends --fix-missing \
        gcc g++ \
        make \
        python3 python3-dev python3-pip python3-venv python3-wheel \
        espeak-ng libsndfile1-dev \
        git \
        wget \
        ffmpeg \
        libsm6 libxext6 \
        libglfw3-dev libgles2-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment (HAMER-style, no conda)
ENV VENV_PATH=/opt/venv
RUN python3 -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

# --------------------------------------------------------------------
# Python / PyTorch / WiLoR deps
# --------------------------------------------------------------------

# Upgrade pip tooling + install PyTorch with CUDA 12.8 wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip wheel setuptools && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install NumPy < 2.0 (for chumpy / smplx compatibility)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "numpy<2.0"

# Install WiLoR + project dependencies
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Fix chumpy compatibility (must be last)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install git+https://github.com/mattloper/chumpy.git --no-build-isolation

# Sapiens requirements
# RUN --mount=type=cache,target=/root/.cache/pip \
#     pip install opencv-python tqdm json-tricks