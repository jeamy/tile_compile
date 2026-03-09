FROM ubuntu:20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        git \
        curl \
        libopencv-dev \
        libeigen3-dev \
        libcfitsio-dev \
        libyaml-cpp-dev \
        nlohmann-json3-dev \
        libssl-dev \
        libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src

RUN cmake -S tile_compile_cpp -B tile_compile_cpp/build-gui2-release -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTS=OFF \
      -DBUILD_QT_GUI=OFF \
    && cmake --build tile_compile_cpp/build-gui2-release -j"$(nproc)"


FROM ubuntu:20.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-pip \
        python3-venv \
        curl \
        libopencv-dev \
        libeigen3-dev \
        libcfitsio-dev \
        libyaml-cpp-dev \
        nlohmann-json3-dev \
        libssl-dev \
        libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/tile_compile

# Backend + frontend
COPY web_backend/app /opt/tile_compile/web_backend/app
COPY web_backend/requirements-backend.txt /opt/tile_compile/web_backend/requirements-backend.txt
COPY web_frontend /opt/tile_compile/web_frontend

# Runtime config/scripts/examples
COPY tile_compile_cpp/examples /opt/tile_compile/tile_compile_cpp/examples
COPY tile_compile_cpp/scripts /opt/tile_compile/tile_compile_cpp/scripts
COPY tile_compile_cpp/tile_compile.yaml /opt/tile_compile/tile_compile_cpp/tile_compile.yaml
COPY tile_compile_cpp/tile_compile.schema.yaml /opt/tile_compile/tile_compile_cpp/tile_compile.schema.yaml
COPY tile_compile_cpp/tile_compile.schema.json /opt/tile_compile/tile_compile_cpp/tile_compile.schema.json

# Built binaries
COPY --from=builder /src/tile_compile_cpp/build-gui2-release/tile_compile_runner /opt/tile_compile/tile_compile_cpp/build/tile_compile_runner
COPY --from=builder /src/tile_compile_cpp/build-gui2-release/tile_compile_cli /opt/tile_compile/tile_compile_cpp/build/tile_compile_cli

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r /opt/tile_compile/web_backend/requirements-backend.txt

# Default runtime paths inside container
RUN mkdir -p /data/input /data/runs /opt/tile_compile/web_backend/runtime

ENV TILE_COMPILE_CLI=/opt/tile_compile/tile_compile_cpp/build/tile_compile_cli
ENV TILE_COMPILE_RUNNER=/opt/tile_compile/tile_compile_cpp/build/tile_compile_runner
ENV TILE_COMPILE_RUNS_DIR=/data/runs
ENV TILE_COMPILE_CONFIG_PATH=/opt/tile_compile/tile_compile_cpp/tile_compile.yaml
ENV TILE_COMPILE_STATS_SCRIPT=/opt/tile_compile/tile_compile_cpp/scripts/generate_report.py
ENV TILE_COMPILE_ALLOWED_ROOTS=/opt/tile_compile:/data/input:/data/runs:/tmp

EXPOSE 8080

CMD ["python3", "-m", "uvicorn", "app.main:app", "--app-dir", "/opt/tile_compile/web_backend", "--host", "0.0.0.0", "--port", "8080"]
