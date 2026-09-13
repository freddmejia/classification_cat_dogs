FROM mambaorg/micromamba:2.9.0-debian13

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml \
    && micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

ARG NV=/opt/conda/lib/python3.12/site-packages/nvidia
ENV LD_LIBRARY_PATH=${NV}/cublas/lib:${NV}/cuda_cupti/lib:${NV}/cuda_nvrtc/lib:${NV}/cuda_runtime/lib:${NV}/cudnn/lib:${NV}/cufft/lib:${NV}/curand/lib:${NV}/cusolver/lib:${NV}/cusparse/lib:${NV}/nccl/lib:${NV}/nvjitlink/lib \
    CUDA_CACHE_MAXSIZE=4294967296

RUN python -c "import numpy, pandas, matplotlib, time, opendatasets, cv2, tensorflow as tf; print('tensorflow', tf.__version__)"

WORKDIR /workspace
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir=/workspace"]
