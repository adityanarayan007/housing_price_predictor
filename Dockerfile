# --- STAGE 1: Dependency Installation ---
# Use a lightweight base image for Python 3.11
FROM python:3.11-slim AS builder

# Set the working directory inside the container
WORKDIR /app

# Copy dependency files
# We only need requirements.txt and setup.py for the final package dependency check
COPY requirements.txt ./
# Copy the setup.py to enable local (non-editable) install of the project code structure
COPY setup.py ./ 

# Install required packages (Production dependencies + Gunicorn/Uvicorn for serving)
# The project's dependencies are installed here.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn uvicorn

# Install the project as a package (using standard install, not editable mode)
# This installs your 'src' package correctly so imports work
RUN pip install --no-cache-dir .

# --- STAGE 2: Final Runtime Environment (The Production Image) ---
# Use a clean, smaller base image for the final deployable container
FROM python:3.11-slim AS final 

# Set the working directory
WORKDIR /app

# Copy the necessary runtime packages and bin files from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy your application code, configuration, and artifacts
COPY src ./src
COPY requirements.txt .
COPY run_training.py .

# Copy artifacts (CRITICAL: Model, scaler, and data files)
COPY artifacts/ artifacts/
COPY data/ data/

# Set the environment variable for the port
ENV PORT=8000 
EXPOSE 8000

# Command to run the application
# Gunicorn manages the workers, and Uvicorn is the ASGI server.
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "src.api.app:app", "--bind", "0.0.0.0:8000"]