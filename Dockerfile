FROM python:3.12-slim AS build
WORKDIR /src
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements-dev.txt
COPY proto ./proto
COPY scripts ./scripts
COPY app ./app
# generate the gRPC stubs into app/ (they are build output, not committed)
RUN PYTHONPATH=/install/lib/python3.12/site-packages python scripts/gen_proto.py

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 PORT=8000 GRPC_PORT=50051 CACHE_DB_PATH=/data/semantic_cache.json
RUN useradd --system --uid 10001 --home /app gateway && mkdir /data && chown gateway /data
WORKDIR /app
COPY --from=build /install /usr/local
COPY --from=build /src/app ./app
USER 10001
VOLUME /data
EXPOSE 8000 50051
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
