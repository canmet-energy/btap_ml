FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /pipelines
COPY requirements.txt /pipelines
RUN pip install -r requirements.txt
COPY src /pipelines
COPY output /output

RUN groupadd -g 1000 app && useradd -u 1000 app -g app

RUN chown -R app:app /pipelines
RUN chown -R app:app /output

USER 1000:1000