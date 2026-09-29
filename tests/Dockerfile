FROM python:3.11-slim

WORKDIR /workspace

COPY tests/requirements.txt /tmp/test-requirements.txt
RUN pip install --no-cache-dir -r /tmp/test-requirements.txt

COPY tests/ /workspace/tests/

# Every service directory, not just frontend. The suite is no longer purely
# black-box: unit tests import service modules directly off the filesystem
# (chunking, model lifecycle, WS decoding, voice naming), and
# test_docker_image_contents.py parses each service's Dockerfile. A missing
# directory here is a collection error, not a skip.
#
# This copies source only — models and datasets are volumes, so the image stays
# small. Keep in sync when a service is added; the >=9 assertion in
# test_docker_image_contents.py fails loudly if one is forgotten.
COPY frontend-service/ /workspace/frontend-service/
COPY stt-service/ /workspace/stt-service/
COPY piper-tts-service/ /workspace/piper-tts-service/
COPY piper-training-service/ /workspace/piper-training-service/
COPY qwen3-tts-service/ /workspace/qwen3-tts-service/
COPY qwen3-asr-service/ /workspace/qwen3-asr-service/
COPY parakeet-asr-service/ /workspace/parakeet-asr-service/
COPY canary-asr-service/ /workspace/canary-asr-service/
COPY chatterbox-tts-service/ /workspace/chatterbox-tts-service/

ENTRYPOINT ["pytest", "-v", "--tb=short", "/workspace/tests"]
