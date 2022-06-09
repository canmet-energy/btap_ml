FROM python:3.8-slim
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
