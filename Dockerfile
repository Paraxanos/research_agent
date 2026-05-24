FROM python:3.11-slim

WORKDIR /app

# Copy everything
COPY . .

# Install backend
RUN cd backend && pip install --no-cache-dir -e .

# Expose port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "research_agent.api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "backend/src"]
