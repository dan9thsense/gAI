FROM python:2.7

RUN apt-get update \
 && apt-get install -y \
 python-pip \
 python-dev \
 && rm -rf /var/lib/apt/lists/*

RUN pip install numpy
RUN pip install tensorflow
RUN pip install matplotlib

WORKDIR /app

COPY agent/ /app/

RUN pip install -r requirements.txt

ENV NAME Learner

CMD ["python", "cloud_socket_connector.py"]

