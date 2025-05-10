FROM python:3.9-slim
# Set the working directory
WORKDIR /app
# Copy the requirements file into the container
COPY . /app
# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

CMD streamlit run app.py --server.address=0.0.0.0 --server.port=8501

