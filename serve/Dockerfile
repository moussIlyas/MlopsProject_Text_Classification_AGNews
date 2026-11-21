# Packaged model image: trained model + artifacts baked in (option C).
# Built at the END of the training job from the fresh artifact/ directory.
# Build: docker build --build-context artifacts=artifacts/ -f serve/Dockerfile -t agnews-model:0.1.0 .
# Run:   docker run agnews-model:0.1.0 "Wall Street stocks rally"
FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Inference-only dependencies, pinned to the training lockfile versions.
COPY serve/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve/predict.py .
# artifact/ is passed as a separate BuildKit context so the training image's
# .dockerignore (which excludes artifact/) does not affect this build.
COPY --from=artifacts . ./artifacts/

ENTRYPOINT ["python", "predict.py"]
