FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ai_workflow_automator.py .
EXPOSE 8080
CMD ["uvicorn", "ai_workflow_automator:app", "--host", "0.0.0.0", "--port", "8080"]
