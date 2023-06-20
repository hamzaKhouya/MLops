# Use a lightweight base image
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy only the necessary files
COPY requirements.txt /app
COPY model/ model/
COPY app.py /app
COPY CICD/ CICD/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables (if required)
ENV ENV_VARIABLE_NAME=example_value

# Remove unnecessary packages and clean the cache
RUN apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Run the script
CMD ["python", "app.py"]
