# Stage 1: Builder Stage
FROM python:3.10-slim AS build

# Set working directory
WORKDIR /app

# Install essential build tools and dependencies
RUN apt-get update && apt-get install -y gcc g++ libffi-dev musl-dev make libgomp1

# Copy only requirements to leverage Docker caching
COPY fastapi/requirements.txt /app/

# Upgrade pip and install Python dependencies and NLTK data
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader stopwords wordnet

# Copy the application code
COPY fastapi/ /app/

# Stage 2: Final Runtime Image
FROM python:3.10-slim AS runtime

# Set working directory
WORKDIR /app

# Install missing dependency (libgomp1) to runtime image
RUN apt-get update && apt-get install -y libgomp1

# Copy installed Python libraries from builder stage
COPY --from=build /app /app/
COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages/
COPY --from=build /usr/local/bin /usr/local/bin/
COPY --from=build /root/nltk_data /root/nltk_data/

# Remove unnecessary files to reduce image size
RUN find /usr/local/lib/python3.10/site-packages -name "*.pyc" -delete \
    && find /usr/local/lib/python3.10/site-packages -name "__pycache__" -type d -exec rm -r {} + 

# Expose the required port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]