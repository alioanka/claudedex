CREATE TABLE IF NOT EXISTS sentiment_logs (
    id SERIAL PRIMARY KEY,
    score NUMERIC,
    source VARCHAR(50),
    metadata JSONB,
    timestamp TIMESTAMP DEFAULT NOW()
);
