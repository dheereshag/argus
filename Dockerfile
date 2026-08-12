# syntax=docker/dockerfile:1

# paddlepaddle has no Linux ARM64 wheels, so the image must target amd64.
# On Apple Silicon this runs under Rosetta/QEMU emulation.
FROM --platform=linux/amd64 python:3.12-slim

# Copy uv from the official uv image (no curl needed).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dependency manifests first for layer caching.
COPY pyproject.toml ./

# Install project + deps (no dev group).
RUN uv sync --no-dev

# Copy the rest of the project (junk excluded via .dockerignore).
COPY . .

# Keep ultralytics config writable inside the container.
ENV YOLO_CONFIG_DIR=/opt/ultralytics
RUN mkdir -p /opt/ultralytics

CMD ["uv", "run", "python", "test_direct.py", "paddleocr"]
