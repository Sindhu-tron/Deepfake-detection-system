# Create API documentation
cat > API_DOCUMENTATION.md << 'EOF'
# API Documentation

## Overview

The Deepfake Detection API provides secure, authenticated access to deepfake detection capabilities with real-time analytics and monitoring.

**Base URL**: `http://localhost:5002`
**Authentication**: API Key required for protected endpoints
**Rate Limiting**: 100 requests per hour per API key

## Authentication

### Request API Key

**Endpoint**: `POST /auth/key`
**Description**: Create a new API key for authentication
**Authentication**: None required

**Request Body**:
```json
{
  "user_name": "Your Name",
  "user_email": "your.email@example.com"  // optional
}