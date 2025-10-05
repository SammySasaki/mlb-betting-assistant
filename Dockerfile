# Use official Python image
FROM python:3.11

# Set working directory
WORKDIR /app
ENV PYTHONPATH=/app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Default command can be overridden by compose
CMD ["python", "infra/db/init_db.py"]