
FROM python:3.14-slim

# Prevent Python from creating .pyc files
# and make logs appear immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install Python dependencies
# COPY requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt

COPY requirements/runtime.txt ./requirements/runtime.txt

RUN pip install --no-cache-dir -r requirements/runtime.txt


# Copy only the application and required ML artifacts
COPY app ./app
COPY src ./src
COPY models ./models

# API port
EXPOSE 8000

# Start FastAPI
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
