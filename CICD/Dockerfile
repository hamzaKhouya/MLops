# Stage 1: Training the LogKeyModel
FROM python:3.9-slim as train

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script and data directory to the working directory
COPY LogKeyModel_train.py .
COPY data ./data

# Run the training script
CMD ["python", "LogKeyModel_train.py"]

# Stage 2: Predicting with the LogKeyModel
FROM python:3.9-slim as predict

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the prediction script, trained model, and data directory to the working directory
COPY LogKeyModel_predict.py .
COPY data ./data

# Run the prediction script
CMD ["python", "LogKeyModel_predict.py"]

# Stage 3: Running visual.py
FROM python:3.9-slim as visual

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY dash_requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r dash_requirements.txt

# Copy the visualization script and data directory to the working directory
COPY visual.py .
COPY data ./data

# Run the visualization script
CMD ["python", "visual.py"]
