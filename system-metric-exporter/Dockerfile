# system_metric-exporter/dockerfile

FROM python:3.10-slim

# Install iostat tool
RUN apt-get update && apt-get install -y sysstat && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY metrics_exporter.py .

# Expose Prometheus scraping port
EXPOSE 18000

# Run the metrics exporter
CMD ["python", "metrics_exporter.py"]
