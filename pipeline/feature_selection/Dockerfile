FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
                
WORKDIR /btap
COPY requirements.txt /btap
RUN pip install -r requirements.txt
COPY feature_selection.py /btap

RUN groupadd -g 1000 app && useradd -u 1000 app -g app

RUN chown -R app:app /btap

USER 1000:1000
CMD ["python", "feature_selection.py"] 