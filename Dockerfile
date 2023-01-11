# Pulls python image from dockerhub
FROM python:3.9

WORKDIR /usr/app

# Adds the relevant files
COPY utils /usr/app/utils
COPY data /usr/app/data
#COPY models /usr/app/models
COPY predictions /usr/app/predictions
COPY main.py /usr/app/
COPY requirements.txt /usr/app/

## Install libs
RUN pip install --no-cache-dir --upgrade pip && \
    pip install -r requirements.txt

# Run the code
CMD ["python", "-u", "utils/evaluate-v2.0.py", "data/dev-v2.0.json", "predictions/preds_deberta-v3-large-squad2.json"]