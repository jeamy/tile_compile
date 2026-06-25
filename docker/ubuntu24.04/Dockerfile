FROM ubuntu:24.04 AS builder

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

RUN cmake -S tile_compile_cpp -B tile_compile_cpp/build-release -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTS=OFF \
    && cmake --build tile_compile_cpp/build-release -j"$(nproc)"

RUN cmake -S web_backend_cpp -B web_backend_cpp/build-release -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTS=OFF \
      -DTILE_COMPILE_BACKEND_STATIC_STDLIB=OFF \
    && cmake --build web_backend_cpp/build-release -j"$(nproc)"

# Build PI AI sidecar (agent_service) – Node.js 22 LTS
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*
RUN cd /src/agent_service && npm install && npm run build


FROM ubuntu:24.04 AS runtime

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

# Node.js 22 LTS for PI AI sidecar
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/tile_compile

# Frontend
COPY web_frontend_v3 /opt/tile_compile/web_frontend_v3

# Runtime config/examples
COPY tile_compile_cpp/examples /opt/tile_compile/tile_compile_cpp/examples
COPY tile_compile_cpp/tile_compile.yaml /opt/tile_compile/tile_compile_cpp/tile_compile.yaml
COPY tile_compile_cpp/tile_compile.schema.yaml /opt/tile_compile/tile_compile_cpp/tile_compile.schema.yaml
COPY tile_compile_cpp/tile_compile.schema.json /opt/tile_compile/tile_compile_cpp/tile_compile.schema.json

# Built binaries
COPY --from=builder /src/tile_compile_cpp/build-release/tile_compile_runner /opt/tile_compile/tile_compile_cpp/build/tile_compile_runner
COPY --from=builder /src/tile_compile_cpp/build-release/tile_compile_cli /opt/tile_compile/tile_compile_cpp/build/tile_compile_cli
COPY --from=builder /src/web_backend_cpp/build-release/tile_compile_web_backend /opt/tile_compile/web_backend_cpp/build/tile_compile_web_backend

# PI AI sidecar (pre-built in builder stage)
COPY --from=builder /src/agent_service/dist /opt/tile_compile/agent_service/dist
COPY --from=builder /src/agent_service/node_modules /opt/tile_compile/agent_service/node_modules
COPY --from=builder /src/agent_service/package.json /opt/tile_compile/agent_service/package.json
COPY --from=builder /src/agent_service/package-lock.json /opt/tile_compile/agent_service/package-lock.json

# Entrypoint script (starts sidecar + backend)
COPY docker/ubuntu24.04/entrypoint.sh /opt/tile_compile/entrypoint.sh
RUN chmod +x /opt/tile_compile/entrypoint.sh

# Default runtime paths inside container
RUN mkdir -p /data/input /data/runs /opt/tile_compile/web_backend_cpp/build

ENV TILE_COMPILE_CLI=/opt/tile_compile/tile_compile_cpp/build/tile_compile_cli
ENV TILE_COMPILE_RUNNER=/opt/tile_compile/tile_compile_cpp/build/tile_compile_runner
ENV TILE_COMPILE_HOST=0.0.0.0
ENV TILE_COMPILE_PORT=8080
ENV TILE_COMPILE_RUNS_DIR=/data/runs
ENV TILE_COMPILE_INPUT_DIR=/data/input
ENV TILE_COMPILE_CONFIG=/opt/tile_compile/tile_compile_cpp/tile_compile.yaml
ENV TILE_COMPILE_SCHEMA=/opt/tile_compile/tile_compile_cpp/tile_compile.schema.yaml
ENV TILE_COMPILE_PRESETS_DIR=/opt/tile_compile/tile_compile_cpp/examples
ENV TILE_COMPILE_UI_DIR=/opt/tile_compile/web_frontend_v3
ENV TILE_COMPILE_PROJECT_ROOT=/opt/tile_compile
ENV TILE_COMPILE_ALLOWED_ROOTS=/opt/tile_compile:/data/input:/data/runs:/tmp
ENV TILE_COMPILE_AGENT_SERVICE_DIR=/opt/tile_compile/agent_service
ENV TILE_COMPILE_AI_AGENT_AUTOSTART=1
ENV AI_AGENT_HOST=127.0.0.1
ENV AI_AGENT_PORT=3001
ENV AI_AGENT_URL=http://127.0.0.1:3001

EXPOSE 8080

CMD ["/opt/tile_compile/entrypoint.sh"]
