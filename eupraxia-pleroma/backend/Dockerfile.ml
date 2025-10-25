FROM python:3.10-slim
WORKDIR /app

# Install system deps for multimedia and rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
	ffmpeg \
	libgl1-mesa-dri \
	libgl1 \
	libglib2.0-0 \
	build-essential \
	libglu1-mesa \
	libx11-6 \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements_ml.txt .
# Install CPU torch wheel separately for reliability
RUN pip install --no-cache-dir torch==2.1.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip install -r requirements_ml.txt --no-cache-dir
RUN pip install --no-cache-dir uvicorn

COPY . /app/backend

# Default to hourly evolution loop. Set EVOLVE_SLEEP_SECONDS in .env to override.
ENV EVOLVE_SLEEP_SECONDS=3600

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & while true; do python /app/backend/evolve_ai.py --examples 200 || true; sleep ${EVOLVE_SLEEP_SECONDS}; done"]
