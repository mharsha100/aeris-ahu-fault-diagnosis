FROM python:3.11.14-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 AERIS_BIND=0.0.0.0 AERIS_DB=/state/ledger.sqlite3
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py /app/
COPY frontend /app/frontend
COPY src /app/src
COPY scripts /app/scripts
COPY tests /app/tests
COPY milestone4 /app/milestone4
COPY experiments/end_to_end_20261116 /app/experiments/end_to_end_20261116
COPY experiments/improvement_20261117 /app/experiments/improvement_20261117
COPY experiments/operator_redesign_20261117/ experiments/operator_redesign_20261117/
COPY experiments/adaptive_lifecycle_20261117/ /app/experiments/adaptive_lifecycle_20261117/
COPY deployment /app/deployment
RUN mkdir -p /state && useradd --uid 10001 --create-home aeris && chown aeris:aeris /state
USER aeris
EXPOSE 8501 8001
HEALTHCHECK --interval=10s --timeout=5s --start-period=90s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/readyz',timeout=4)"
CMD ["python","deployment/serve.py"]
