# Docker Build

`Dockerfile.cuda12.8-runtime` is the release Dockerfile.

Published image:

```bash
docker pull jaceju68/glm-asr-stack:cuda12.8-runtime
```

Build locally:

```bash
docker build --network=host \
  -f docker/Dockerfile.cuda12.8-runtime \
  -t glm-asr-stack:cuda12.8-runtime .
```
