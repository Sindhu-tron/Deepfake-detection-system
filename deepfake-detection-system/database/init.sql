-- Database initialization for deepfake detection system

-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Predictions table
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    image_hash VARCHAR(64) NOT NULL,
    prediction_class VARCHAR(10) NOT NULL,
    confidence FLOAT NOT NULL,
    real_probability FLOAT NOT NULL,
    fake_probability FLOAT NOT NULL,
    processing_time_ms FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address INET
);

-- Insert default user
INSERT INTO users (username, email) VALUES 
('demo', 'demo@example.com');