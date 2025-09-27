#!/bin/bash
# AWS Deployment Script

set -e

echo "🚀 Deploying Deepfake Detection to AWS"
echo "====================================="

# Variables
APP_NAME="deepfake-detection"
REGION="us-west-2"
INSTANCE_TYPE="t3.medium"
KEY_NAME="your-key-pair"

# Create EC2 instance
echo "📦 Creating EC2 instance..."
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --count 1 \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-groups default \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$APP_NAME}]" \
    --user-data file://user-data.sh

echo "✅ Deployment initiated. Check AWS console for status."