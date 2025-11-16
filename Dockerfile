FROM nvidia/cuda:12.6.0-cudnn9-runtime-ubuntu22.04

# Set Python version
ENV PYTHON_VERSION=3.12
ENV APP_HOME=/app
WORKDIR $APP_HOME

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser $APP_HOME
USER appuser

# Expose the port Cloud Run uses
EXPOSE 8080

# Use uvicorn with production config
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
