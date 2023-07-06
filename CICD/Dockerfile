# Use a lightweight base image
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy only the necessary files
COPY requirements.txt /app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Remove unnecessary packages and clean the cache
RUN apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Run the script
CMD ["python", "visual.py"]
