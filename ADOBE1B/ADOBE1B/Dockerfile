# Use a slim Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app



# Copy requirements file and install dependencies
COPY requirements.txt .
# The --no-cache-dir flag is used to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# --- Model Caching ---
# This is the crucial step for offline execution.
# We run a Python script at build time to download and cache the models.
COPY cache_models.py .
RUN python cache_models.py

# Copy the application source code
COPY main_code.py .

# Create directories for input and output
RUN mkdir -p /app/input /app/output

# Define the command to run the application
# It will expect input files in /app/input and write output to /app/output
ENTRYPOINT ["python", "main_code.py"]
