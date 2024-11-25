#!/bin/bash

echo "Downloading environment variables..."
aws s3 cp s3://ytcodedeploybucket/dagshub.env /home/ubuntu/dagshub.env