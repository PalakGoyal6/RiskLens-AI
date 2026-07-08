# Use Python 3.12 slim base image for a smaller memory footprint
FROM python:3.12-slim

# Install system dependencies required for lightgbm, xgboost, or catboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose port 8000 (though Render dynamically binds to $PORT)
EXPOSE 8000

# Start Uvicorn with a single worker to fit in Render's 512MB RAM free tier
CMD uvicorn server:app --host 0.0.0.0 --port $PORT --workers 1
