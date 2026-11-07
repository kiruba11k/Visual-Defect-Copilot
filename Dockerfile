FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
COPY app ./app
RUN pip install --no-cache-dir .
COPY . .
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
