FROM python:3.9

# Set working directory inside the container
WORKDIR /app

COPY requirements.txt .

# Create a virtual environment
RUN python -m venv venv

# Activate the virtual environment and install dependencies
RUN . venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code into the container
COPY . .

# Expose port 8000 for FastAPI application
EXPOSE 8000

# Command to run the FastAPI application using uvicorn
CMD ["./venv/bin/uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
