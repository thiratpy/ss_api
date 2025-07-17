# Use an official, stable PyTorch base image
FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

# Install pip, git, and all the required system libraries for OpenCV
RUN apt-get update && apt-get install -y python3-pip git libgl1-mesa-glx libglib2.0-0

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt --timeout=300

# Copy the rest of your application code
COPY . .

# Command to run your worker when the container starts
CMD ["python", "-u", "runpod_worker.py"]