FROM python:3.13-slim

WORKDIR /srv

COPY app/backend/requirements.txt app/backend/requirements.txt
RUN pip install --no-cache-dir -r app/backend/requirements.txt

COPY app/backend/ app/backend/
COPY app/__init__.py app/__init__.py

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
