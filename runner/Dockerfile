# Self-hosted GitHub Actions runner for the AG News training pipeline.
# Base image already ships git, curl, jq, gcc and the runner agent.
FROM myoung34/github-runner:latest

USER root

# The project requires Python 3.13 (base image has 3.14). Install uv and let
# uv manage the exact interpreter; jobs then run `uv sync --frozen`.
RUN curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR=/usr/local/bin sh \
    && uv --version \
    && uv python install 3.13 \
    && uv python list

# Keep the base image ENTRYPOINT/CMD: it registers the runner and starts it.
