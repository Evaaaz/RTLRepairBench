# syntax=docker/dockerfile:1.7
# The Ubuntu digest is a multi-platform 24.04 image index.  Every CAD binary is
# supplied by the checksum-pinned official YosysHQ OSS CAD Suite archive below.
FROM ubuntu@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90

ARG TARGETARCH
ARG OSS_CAD_SUITE_TAG=2026-05-08
ARG OSS_CAD_SUITE_RELEASE_ID=20260508
ARG OSS_ARCH=linux-arm64
ARG OSS_SHA256=4b056c3e999c4289db22decb9e2be06178e9c66d6f56e48d8f1c9c85d1452f72

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# BuildKit verifies the remote artifact before exposing it to the build.  The
# wrapper script supplies the architecture-specific pair from toolchain.lock.
ADD --checksum=sha256:${OSS_SHA256} \
    https://github.com/YosysHQ/oss-cad-suite-build/releases/download/${OSS_CAD_SUITE_TAG}/oss-cad-suite-${OSS_ARCH}-${OSS_CAD_SUITE_RELEASE_ID}.tgz \
    /tmp/oss-cad-suite.tgz

# The portable Perl launcher expects host Perl's FindBin, which is absent in
# the intentionally package-free base.  The bundled native binary is the same
# pinned Verilator build and needs no host modules.
RUN case "${TARGETARCH}:${OSS_ARCH}" in \
      amd64:linux-x64|arm64:linux-arm64) ;; \
      *) echo "TARGETARCH=${TARGETARCH} disagrees with OSS_ARCH=${OSS_ARCH}" >&2; exit 2 ;; \
    esac \
    && tar -xzf /tmp/oss-cad-suite.tgz -C /opt \
    && rm /tmp/oss-cad-suite.tgz \
    && ln -sf verilator_bin /opt/oss-cad-suite/bin/verilator

ENV PATH="/opt/oss-cad-suite/bin:/opt/oss-cad-suite/py3bin:${PATH}" \
    VERILATOR_ROOT=/opt/oss-cad-suite/share/verilator \
    GHDL_PREFIX=/opt/oss-cad-suite/lib/ghdl \
    HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/work/code

RUN yosys -V \
    && sby --version \
    && iverilog -V \
    && verilator --version \
    && z3 --version \
    && boolector --version

WORKDIR /work
# The standalone repository keeps the verifier under code/benchmarks. The
# source tree is mounted read-only/read-write by the wrapper at /work; the
# image intentionally contains only the pinned toolchain.
ENTRYPOINT ["python3", "/work/code/benchmarks/formal_verify.py"]
CMD ["--help"]
