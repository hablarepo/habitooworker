# Habitoo worker: video -> Gaussian splat. Běží na NVIDIA GPU (WSL2 na Windows i cloud).
# V obrazu NENÍ žádné tajemství: klíč přichází až za běhu přes env_file/proměnné (viz docker-compose.yml).
# "devel" základ kvůli nvcc – gsplat se kompiluje už při buildu (chyba shodí build, ne první úlohu).
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Architektury GPU, pro které se gsplat zkompiluje.
# 7.5 = GTX 1660 Super (Turing). "+PTX" umožní běh i na novějších GPU (cloud) bez změny obrazu,
# jen s pomalejším prvním startem. [DOPLNIT] pro konkrétní cloudovou GPU lze přidat její architekturu,
# např. --build-arg CUDA_ARCHS="7.5;8.6+PTX".
ARG CUDA_ARCHS="7.5+PTX"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential ninja-build \
      libglib2.0-0 libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# PyTorch s CUDA 12.1 (kompatibilní s ovladači NVIDIA >= 525).
RUN pip install --upgrade pip setuptools wheel \
 && pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# gsplat se musí stavět bez izolace (potřebuje nainstalovaný torch). MAX_JOBS omezuje paralelní
# kompilaci, aby build nespolkl RAM (16 GB PC s WSL2).
RUN pip install ninja jaxtyping rich \
 && TORCH_CUDA_ARCH_LIST="${CUDA_ARCHS}" MAX_JOBS=2 \
    pip install --no-build-isolation gsplat==1.5.3

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
RUN pip install . \
 && python3 -c "import habitoo_worker, cv2, pycolmap, gsplat; print('import OK')"

# Běh pod běžným uživatelem; /work je pracovní složka (mazaná po každé úloze).
RUN useradd -m -u 1000 worker && mkdir -p /work /out && chown -R worker:worker /work /out /app
USER worker
ENV WORK_DIR=/work

CMD ["python3", "-m", "habitoo_worker.main"]
