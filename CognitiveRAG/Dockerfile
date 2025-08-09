# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./CognitiveRAG/requirements.txt /app/

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY ./CognitiveRAG /app/CognitiveRAG

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

# Run main.py when the container launches
# The command will be something like: uvicorn CognitiveRAG.main:app --host 0.0.0.0 --port 8000
# We will use a main_server.py as entrypoint for the server.
CMD ["uvicorn", "CognitiveRAG.main_server:app", "--host", "0.0.0.0", "--port", "8000"]
