FROM python:3.12.6-slim

WORKDIR /app

# 1. Upgrade pip inside the container to handle large file streams better
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .

# 2. Install all small packages first (Docker will cache this step)
RUN grep -v "tensorflow" requirements.txt > requirements_sub.txt && \
    pip install --no-cache-dir -r requirements_sub.txt

# 3. Install TensorFlow from a locally pre-downloaded wheel (avoids flaky long-stream downloads)
COPY tf_wheels/ /tmp/tf_wheels/
RUN pip install --no-cache-dir /tmp/tf_wheels/tensorflow-2.21.0-*.whl

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
