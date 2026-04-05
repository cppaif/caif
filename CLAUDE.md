# CAIF Public — GitHub Release Repository

This is the public release repo. Do NOT develop here.

## Workflow

- **Never edit this repo directly** — all development happens in `../caif/`
- Changes are merged from the private `caif` repo when ready for release
- Build system: **CMake only** (no SConstruct)
- No context/tracking files (CHANGES.md, debug docs) belong here

## GitHub

- Remote: https://github.com/cppaif/caif
- Branch `github-release` pushes to `main` on GitHub
- Push: `git push github github-release:main`

## Build

```
mkdir build && cd build
cmake .. \
  -DOPENBLAS_INCLUDE_DIR=/mnt/s/dev/ise/third_party/include \
  -DOPENBLAS_LIB_DIR=/mnt/s/dev/ise/third_party/lib/release/linux \
  -DEIGEN3_INCLUDE_DIR=/mnt/s/dev/ise/third_party/include \
  -DCUDA_INCLUDE_DIR=/mnt/s/dev/ise/third_party/include \
  -DCUDA_LIB_DIR=/mnt/s/dev/ise/third_party/lib/release/linux/cuda
make -j$(nproc)
```
