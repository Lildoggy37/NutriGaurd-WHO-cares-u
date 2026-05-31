FROM python:3.12-slim-bookworm

WORKDIR /app

# 系统依赖：gcc 用于编译 torch/sentence-transformers 的 C 扩展
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖（分层缓存）
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# 应用代码
COPY . .

# 国内镜像加速（可按需设为 hf-mirror.com）
ENV HF_ENDPOINT=https://huggingface.co

# 默认值，docker-compose 会覆盖
ENV REDIS_HOST=redis

EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
