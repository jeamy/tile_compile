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
    && cmake --build tile_compile_cpp/build-gui2-release -j"$(nproc)"

RUN cmake -S web_backend_cpp -B web_backend_cpp/build-release -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTS=OFF \
      -DTILE_COMPILE_BACKEND_STATIC_STDLIB=OFF \
    && cmake --build web_backend_cpp/build-release -j"$(nproc)"


FROM ubuntu:20.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
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

# Frontend
COPY web_frontend /opt/tile_compile/web_frontend

# Runtime config/examples
COPY tile_compile_cpp/examples /opt/tile_compile/tile_compile_cpp/examples
COPY tile_compile_cpp/tile_compile.yaml /opt/tile_compile/tile_compile_cpp/tile_compile.yaml
COPY tile_compile_cpp/tile_compile.schema.yaml /opt/tile_compile/tile_compile_cpp/tile_compile.schema.yaml
COPY tile_compile_cpp/tile_compile.schema.json /opt/tile_compile/tile_compile_cpp/tile_compile.schema.json

# Built binaries
COPY --from=builder /src/tile_compile_cpp/build-gui2-release/tile_compile_runner /opt/tile_compile/tile_compile_cpp/build/tile_compile_runner
COPY --from=builder /src/tile_compile_cpp/build-gui2-release/tile_compile_cli /opt/tile_compile/tile_compile_cpp/build/tile_compile_cli
COPY --from=builder /src/web_backend_cpp/build-release/tile_compile_web_backend /opt/tile_compile/web_backend_cpp/build/tile_compile_web_backend

# Default runtime paths inside container
RUN mkdir -p /data/input /data/runs /opt/tile_compile/web_backend_cpp/build

ENV TILE_COMPILE_CLI=/opt/tile_compile/tile_compile_cpp/build/tile_compile_cli
ENV TILE_COMPILE_RUNNER=/opt/tile_compile/tile_compile_cpp/build/tile_compile_runner
ENV TILE_COMPILE_HOST=0.0.0.0
ENV TILE_COMPILE_PORT=8080
ENV TILE_COMPILE_RUNS_DIR=/data/runs
ENV TILE_COMPILE_CONFIG=/opt/tile_compile/tile_compile_cpp/tile_compile.yaml
ENV TILE_COMPILE_SCHEMA=/opt/tile_compile/tile_compile_cpp/tile_compile.schema.yaml
ENV TILE_COMPILE_PRESETS_DIR=/opt/tile_compile/tile_compile_cpp/examples
ENV TILE_COMPILE_UI_DIR=/opt/tile_compile/web_frontend
ENV TILE_COMPILE_PROJECT_ROOT=/opt/tile_compile
ENV TILE_COMPILE_ALLOWED_ROOTS=/opt/tile_compile:/data/input:/data/runs:/tmp

EXPOSE 8080

CMD ["/opt/tile_compile/web_backend_cpp/build/tile_compile_web_backend"]
