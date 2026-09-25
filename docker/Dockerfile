# MicroHH GPU build (single precision, CUDA, Ampere 86 = RTX 30 series) for sim-hub. Code: MHH.
# Build context is the repository root:  docker build -f docker/Dockerfile -t ghcr.io/ngc4261/sim-mhh:2026.09 .
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gfortran cmake git ca-certificates \
        libnetcdf-dev libfftw3-dev libboost-dev libhdf5-dev \
        python3 python3-numpy python3-netcdf4 python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# rte-rrtmgp-cpp (required by CMakeLists.txt) and kernel_launcher are https submodules.
RUN git submodule update --init --recursive

# Same configuration as the RTX 3090 host build that ran weisman_klemp 768x768x192 for 3 h
# (2026-09-23): SYST=ubuntu_20lts (sets CUDA arch 86), USECUDA, single precision.
WORKDIR /app/build
RUN cmake .. -DSYST=ubuntu_20lts -DUSECUDA=TRUE -DUSESP=TRUE -DCMAKE_BUILD_TYPE=RELEASE \
    && make -j"$(nproc)" \
    && ls -la /app/build/microhh

# Launcher: writes the initial profiles, sets pbot, runs init, adds the warm bubble, runs the case.
# No ENTRYPOINT: sim-hub passes the full command (`docker run <image> <command...>`).
COPY docker/run_wk.sh /app/run_wk.sh
RUN chmod +x /app/run_wk.sh

CMD ["/app/run_wk.sh"]
