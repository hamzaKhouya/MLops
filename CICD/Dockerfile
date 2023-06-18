# Use a base image with the required dependencies
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script and model files to the working directory
COPY . .

# Set the command to run when the container starts
CMD ["python", "app.py"]
#CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8050"]