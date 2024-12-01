# Use a lightweight Python image
FROM python:3.12-slim

WORKDIR /usr/src/app

# Install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the model and app directories
COPY model/linear_regression_pricing_model.pkl model/linear_regression_pricing_model.pkl
COPY app app

# Expose the port that Gradio will run on
EXPOSE 7860

# Ensure Gradio listens on all network interfaces
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Command to run the Gradio app
CMD ["python", "app/app.py"]

