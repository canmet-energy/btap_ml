FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
                
WORKDIR /pipelines
COPY requirements.txt /pipelines
RUN pip install -r requirements.txt
COPY ../../src /pipelines/component/src
# COPY ../src/preprocessing.py /btap-project

RUN groupadd -g 1000 app && useradd -u 1000 app -g app

RUN chown -R app:app /pipelines

USER 1000:1000
# CMD ["python", "../src/preprocessing.py"] 