# Dockerfile

# --- Build Stage ---
FROM python:3.12-slim as builder

WORKDIR /code

COPY ./backend/requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt


# --- Final Stage ---
FROM python:3.12-slim

WORKDIR /code

# Copy the installed packages (libraries) from the build stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy the executables (like uvicorn) from the build stage's bin directory
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code from your backend folder into the container
COPY ./backend/ .

# Expose port 8000 to allow communication to the container
EXPOSE 8000

# Define the command to run your app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
