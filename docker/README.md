# Docker Build

`Dockerfile.cuda12.8-runtime` is the full CUDA runtime release Dockerfile.
`Dockerfile.cuda12.8-slim` is the smaller image that keeps only cudart,
cuBLAS/Lt, and cuFFT.

Published image:

```bash
docker pull jaceju68/glm-asr-stack:cuda12.8-slim
```

Build the slim image locally:

```bash
docker build --network=host \
  -f docker/Dockerfile.cuda12.8-slim \
  -t glm-asr-stack:cuda12.8-slim .
```

Build the full CUDA runtime image:

```bash
docker build --network=host \
  -f docker/Dockerfile.cuda12.8-runtime \
  -t glm-asr-stack:cuda12.8-runtime .
```
