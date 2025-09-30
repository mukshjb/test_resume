# Use the official TensorFlow GPU image as the base image
FROM tensorflow/tensorflow:2.14.0-gpu

# Set the working directory inside the container
WORKDIR /tf/code

# Copy the current directory (project directory) to the container's working directory
COPY . /tf/code

# Install any necessary dependencies (like matplotlib)
RUN pip install --no-cache-dir -r /tf/code/requirements.txt

# Install any additional system dependencies if necessary
# RUN apt-get update && apt-get install -y <required-packages>



# Run the Python script (train_transfer_learning.py) when the container starts
CMD ["python", "/tf/code/train_transfer_learning.py"]