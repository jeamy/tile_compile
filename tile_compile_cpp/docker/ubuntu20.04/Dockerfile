FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        cmake \
        git \
        curl \
        wget \
        file \
        zip \
        pkg-config \
        patchelf \
        libgl1-mesa-dev \
        libopencv-dev \
        libeigen3-dev \
        libcfitsio-dev \
        libyaml-cpp-dev \
        nlohmann-json3-dev \
        libssl-dev \
        qt6-base-dev \
        qt6-base-dev-tools \
        libqt6printsupport6 \
        qt6-tools-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
