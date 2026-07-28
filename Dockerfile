FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

COPY src/ ./src/
COPY tests/ ./tests/

EXPOSE 5000

CMD ["pytest"]
