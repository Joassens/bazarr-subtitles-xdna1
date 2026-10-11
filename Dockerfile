# npusubs: subtitle-grade Whisper on the AMD XDNA1 NPU.
#
# BASE must be the image built from https://github.com/drakosha/whisper-xdna with
#   docker build --build-arg WITH_REFERENCE=1 -t npu-whisper:latest .
# (it carries XRT, MLIR-AIE/IRON, peano and the torch + openai-whisper environment).
ARG BASE=npu-whisper:latest
FROM ${BASE}

# Extras for the HTTP API, VAD and (optional) speaker diarization.
RUN /opt/npu-whisper/refenv/bin/pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      fastapi "uvicorn[standard]" python-multipart torchaudio silero-vad "pyannote.audio>=4"

COPY npusubs /opt/npusubs/npusubs
COPY subcfg/subcfg.py /opt/npusubs/subcfg.py
COPY config.example/schemas /opt/npusubs/schemas

ENV PYTHONPATH=/usr/lib/python3/dist-packages:/opt/npusubs \
    PEANO_INSTALL_DIR=/opt/mlir-aie/ironenv/lib/python3.13/site-packages/llvm-aie \
    XDNA_ROOT=/opt/npu-whisper \
    SUBCFG_DIR=/config \
    NPUSUBS_HOST=0.0.0.0 \
    NPUSUBS_PORT=9091

WORKDIR /opt/npu-whisper
EXPOSE 9091
CMD ["/opt/npu-whisper/refenv/bin/python", "-m", "npusubs"]
