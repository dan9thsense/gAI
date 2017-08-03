FROM python:3.5.3-slim

WORKDIR /app

ADD agent/* /app/
ADD agent/learners/* /app/learners/

RUN pip install -r requirements.txt

ENV NAME Learner

CMD ["python", "example_learner.py"]
