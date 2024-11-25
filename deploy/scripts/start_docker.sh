#!/bin/bash

# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Starting deployment process..."

# Load DAGSHUB_PAT from the environment file
if [ -f /home/ubuntu/dagshub.env ]; then
  echo "Loading environment variables from /home/ubuntu/dagshub.env..."
  export $(cat /home/ubuntu/dagshub.env | xargs)
else
  echo "Environment file not found! Deployment cannot proceed."
  exit 1
fi

# Verify DAGSHUB_PAT is loaded
if [ -z "$DAGSHUB_PAT" ]; then
  echo "DAGSHUB_PAT is not set. Exiting..."
  exit 1
fi
echo "Environment variables loaded successfully."

# Log in to Amazon ECR
echo "Logging in to ECR..."
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/m3t3s7a1
if [ $? -ne 0 ]; then
  echo "Failed to log in to ECR. Exiting..."
  exit 1
fi

# Pull the latest Docker image
echo "Pulling Docker image..."
docker pull public.ecr.aws/m3t3s7a1/yt-plugin:latest
if [ $? -ne 0 ]; then
  echo "Failed to pull Docker image. Exiting..."
  exit 1
fi

# Check and remove any existing container
CONTAINER_NAME="yt-plugin"
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
  echo "Stopping existing container..."
  docker stop $CONTAINER_NAME
fi

if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
  echo "Removing existing container..."
  docker rm $CONTAINER_NAME
fi

# Run the new container
echo "Starting new container..."
docker run -d -p 80:5000 --name yt-plugin -e DAGSHUB_PAT=$DAGSHUB_PAT public.ecr.aws/m3t3s7a1/yt-plugin:latest
if [ $? -ne 0 ]; then
  echo "Failed to start Docker container. Exiting..."
  exit 1
fi

echo "Container started successfully."

# Clean up the environment file
echo "Cleaning up environment file..."
rm -f /home/ubuntu/dagshub.env

echo "Deployment process completed successfully."