FROM python:3.12-slim-bookworm

WORKDIR /app

# 清华 pip 镜像
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

COPY requirements-docker.txt .
RUN pip install --no-cache-dir --default-timeout=600 -r requirements-docker.txt

COPY . .

ENV REDIS_HOST=redis
EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
