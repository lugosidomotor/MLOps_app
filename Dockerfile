# Stage 1: Training
FROM python:3.8-slim as builder

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Run the training script
RUN python train.py

# Stage 2: Final image for serving the model
FROM python:3.8-slim

LABEL maintainer="Umpa Lumpa <dik@duk.com>"

ENV USER=serviceuser
RUN adduser -D $USER
USER $USER

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trained model from the builder stage
COPY --from=builder /app/model.pth /app/model.pth

# Copy the necessary application files
COPY app.py /app/app.py
COPY train.py /app/train.py

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
