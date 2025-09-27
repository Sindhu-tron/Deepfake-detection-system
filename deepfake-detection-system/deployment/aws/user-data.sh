#!/bin/bash
# EC2 User Data Script

# Update system
yum update -y

# Install Docker
yum install -y docker
systemctl start docker
systemctl enable docker

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Add ec2-user to docker group
usermod -a -G docker ec2-user

# Create application directory
mkdir -p /app
cd /app

# Clone or copy application files
# (You would copy your docker-compose.prod.yml and other files here)

# Start services
docker-compose -f docker-compose.prod.yml up -d

echo "Deepfake Detection deployed successfully"