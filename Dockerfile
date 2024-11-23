FROM python:3.10-slim

# Install build dependencies (gcc, make, etc.)
RUN apt-get update && apt-get install -y \
    gcc \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY fastapi/ /app/

RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD ["python", "app.py"]