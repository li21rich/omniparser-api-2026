FROM registry.hf.space/microsoft-omniparser:latest

USER root

RUN chmod 1777 /tmp \
    && apt update -q && apt install -y ca-certificates wget libgl1 \
    && wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb && apt update -q \
    && apt install -y --no-install-recommends libcudnn8 libcublas-12-2

RUN pip install --no-cache-dir fastapi[all] transformers==4.38.2 \
    paddleocr==2.10.0 paddlepaddle==3.0.0 torch==2.5.1 torchvision==0.20.1 numpy==1.26.4


COPY main.py main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]