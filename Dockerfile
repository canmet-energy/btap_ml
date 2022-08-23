# Use the tensorflow gpu base image to allow gpu support
FROM tensorflow/tensorflow:2.10.0rc0-gpu

# Install and setup target Python versions
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip
RUN python3 -m pip --no-cache-dir install --upgrade \
    "pip<20.3" \
    setuptools

# Create the working directory
WORKDIR /home/btap_ml

# Copy the reuirements and Python files
COPY requirements.txt ./
COPY src ./src
# Install the requirements
RUN pip install --no-cache-dir -r requirements.txt
# Create the input and output directories
RUN mkdir output
RUN mkdir input
# Start the process
CMD ["/bin/bash"]
