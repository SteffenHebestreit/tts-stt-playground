FROM python:3.11-slim

WORKDIR /workspace

COPY tests/requirements.txt /tmp/test-requirements.txt
RUN pip install --no-cache-dir -r /tmp/test-requirements.txt

COPY tests/ /workspace/tests/
COPY frontend-service/ /workspace/frontend-service/

ENTRYPOINT ["pytest", "-v", "--tb=short", "/workspace/tests"]
