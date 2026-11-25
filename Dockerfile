FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

COPY config/ config/
COPY prompts/ prompts/
COPY data/sample_fec.txt data/sample_fec.txt

RUN useradd --create-home appuser
USER appuser
EXPOSE 8080

CMD ["sh", "-c", "uvicorn audit_copilot.api:create_app --factory --host 0.0.0.0 --port ${PORT:-8080}"]
