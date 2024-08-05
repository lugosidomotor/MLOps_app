# Stage 1: Data Preparation
FROM python:3.8-slim as prep

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the data preparation script and run it
COPY /helloapp/data_prep.py data_prep.py
RUN python data_prep.py

# Stage 2: Training
FROM python:3.8-slim as train

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy data preparation output and other necessary files
COPY --from=prep /app/train.csv /app/train.csv
COPY --from=prep /app/test.csv /app/test.csv
COPY /helloapp .

# Run the training script
RUN python train.py

# Stage 3: Evaluation
FROM python:3.8-slim as evaluate

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy training output and other necessary files
COPY --from=train /app/model.pth /app/model.pth
COPY --from=train /app/train.csv /app/train.csv
COPY --from=train /app/test.csv /app/test.csv
COPY /helloapp .

# Run the evaluation script
RUN python evaluate.py

# Stage 4: Final image for serving the model
FROM python:3.8-slim

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trained model and necessary files from the evaluate stage
COPY --from=evaluate /app/model.pth /app/model.pth
COPY /helloapp .

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]