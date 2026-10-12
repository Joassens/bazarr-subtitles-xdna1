# The bridge Bazarr talks to: the upstream image (Python environment) plus this repository's modified bridge and web UI.
# Build context is the repository root (see ../compose.yml):  docker build -f bridge/Dockerfile .
FROM mccloud/bazarr-openai-whisperbridge

COPY bridge/bridge.py /app/bazarr-openai-whisperbridge.py
COPY subcfg/subcfg.py /app/subcfg.py
COPY bridge/ui /app/ui

ENV SUBCFG_DIR=/config
EXPOSE 9000
