#!/usr/bin/env bash

python run.py tasks_config.challenge.json -l learners.base.RemoteLearner
python local_socket_connector.py
