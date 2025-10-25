FROM python:3.11-slim

# Make pip more resilient to slow networks during docker build
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy requirements and upgrade pip/setuptools/wheel before installing
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
	&& pip install --no-cache-dir -r requirements.txt --timeout 120 --retries 5 --prefer-binary

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]