# syntax=docker/dockerfile:1.4
# Dockerfile for CHAMPSS Timing Pipeline Web Service

# Base image
FROM python:3.10-slim-bullseye

# Set the working directory and copy necessary files
WORKDIR /app

# Install basic dependencies and upgrade system packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends git openssh-client build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies and clone the champss_timing repository
RUN pip3 install --no-cache-dir numpy scipy requests matplotlib flask slack_bolt astropy gitpython; \
    git clone --depth 1 https://github.com/chime-sps/champss_timing.git /app/champss_timing

# Expose the application port
EXPOSE 5000

# Add github into the known hosts to avoid SSH warnings
RUN mkdir -p /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts

# Set the environment variable and start the application
CMD sh -c "chmod 400 /root/.ssh/id_ed25519; python3 champss_timing server -r $GITHUB_REPO_URL --ssh-key /root/.ssh/id_ed25519 --password $PASSWORD --slack $SLACK_CHANNEL --root $WEB_ROOT -p $WEB_PORT --host 0.0.0.0"