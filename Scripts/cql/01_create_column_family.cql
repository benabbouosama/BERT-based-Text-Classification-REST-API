CREATE TABLE IF NOT EXISTS textclassifier.text_data (
    id UUID PRIMARY KEY,
    text TEXT,
    label INT,
    probability FLOAT,
    username TEXT,
    timestamp TIMESTAMP
);
