# Use the official Python image from the Docker Hub
FROM python

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container at /app
COPY . /app/

# Copy the SQLite database file into the container
COPY database.db /app/

# Expose port 80 for the FastAPI application
EXPOSE 80

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
