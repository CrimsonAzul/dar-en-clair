FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
COPY src ./src
COPY config ./config
# torch CPU : le conteneur n'a pas de GPU, et les roues CUDA pesent plusieurs Go
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir ".[ml]"
ENV FISCAL_CONFIG=/app/config/fiscal_2026.yaml \
    RECHERCHE_CONFIG=/app/config/recherche.yaml \
    ADAPTATEUR=/app/models/bge-m3-lora
EXPOSE 8000
CMD ["uvicorn", "darenclair.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
