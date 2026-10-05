#!/usr/bin/env bash
set -euo pipefail

# Standalone bundle root: this script lives at <repo>/code/scripts/.  A nested
# checkout can still select its outer repository explicitly with RTLREPAIR_ROOT.
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${RTLREPAIR_ROOT:-${BUNDLE_ROOT}}"
IMAGE="${RTLREPAIR_FORMAL_IMAGE:-rtlrepair-formal:20260508}"

case "$(uname -m)" in
  arm64|aarch64)
    PLATFORM="linux/arm64"
    DOCKER_TARGETARCH="arm64"
    OSS_ARCH="linux-arm64"
    OSS_SHA256="4b056c3e999c4289db22decb9e2be06178e9c66d6f56e48d8f1c9c85d1452f72"
    ;;
  x86_64|amd64)
    PLATFORM="linux/amd64"
    DOCKER_TARGETARCH="amd64"
    OSS_ARCH="linux-x64"
    OSS_SHA256="c71735b02df363e2ad8e9f129e477e4f31b18d6532e9bed92eb8ad296101cb6f"
    ;;
  *) echo "Unsupported host architecture: $(uname -m)" >&2; exit 2 ;;
esac

if [[ "${1:-}" == "--build" ]]; then
  FORCE_BUILD=1
  shift
else
  FORCE_BUILD=0
fi

if [[ "${FORCE_BUILD}" == "1" ]] || ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  docker build --platform "${PLATFORM}" --tag "${IMAGE}" \
    --build-arg "TARGETARCH=${DOCKER_TARGETARCH}" \
    --build-arg "OSS_ARCH=${OSS_ARCH}" \
    --build-arg "OSS_SHA256=${OSS_SHA256}" \
    --file "${ROOT}/docker/formal/Dockerfile" "${ROOT}/docker/formal"
fi

IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${IMAGE}")"

docker run --rm --platform "${PLATFORM}" \
  --user "$(id -u):$(id -g)" \
  --env "RTLREPAIR_FORMAL_IMAGE=${IMAGE}" \
  --env "RTLREPAIR_FORMAL_IMAGE_ID=${IMAGE_ID}" \
  --volume "${ROOT}:/work" \
  --workdir /work \
  "${IMAGE}" "$@"
