# Use a minimal Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install only necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first (for caching)
COPY requirements.txt /app/

# Install dependencies efficiently
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip 
# Remove cache after install

# Copy only necessary project files
COPY . /app

# Expose the Flask application port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=crypto_app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=development

# Run the Flask app
CMD ["python", "crypto_app.py"]
