FROM python:3.11-slim

# Install system deps for GeoPandas, PostGIS, and GDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL env
ENV GDAL_CONFIG=/usr/bin/gdal-config

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download Natural Earth coastline data for Stage 14
RUN mkdir -p /app/data/coastline && \
    curl -sL "https://naciscdn.org/naturalearth/110m/physical/ne_110m_coastline.zip" \
    -o /tmp/coastline.zip && \
    python -m zipfile -e /tmp/coastline.zip /app/data/coastline/ && \
    rm /tmp/coastline.zip

# Download Natural Earth land polygons for landfall detection
RUN curl -sL "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip" \
    -o /tmp/land.zip && \
    python -m zipfile -e /tmp/land.zip /app/data/coastline/ && \
    rm /tmp/land.zip

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
