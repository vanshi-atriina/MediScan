FROM python:3.10-slim

# Install system packages needed for TLS/SSL and PDF processing
RUN apt-get update && apt-get install -y \
    gcc \
    libssl-dev \
    ca-certificates \
    curl \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8004

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
