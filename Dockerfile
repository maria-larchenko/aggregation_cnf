# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster
WORKDIR /aggregation_app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY ./aggregation ./aggregation
COPY ./datasets/dataset_S10_shape ./aggregation/dataset_S10_shape
COPY ./output_2d/shape/4_II ./output_2d/shape/4_II
COPY aggregation_cnf.py aggregation_cnf.py
CMD ["python3", "aggregation_cnf.py"]
