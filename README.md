# si-project
Files:
- utils/evaluate-v2.0.py: Function evaluation already done. It is the official evaluation.
- data/dev-v2.0.json: Our test file. Answering these questions, we obtain the evaluation
- data/train-v2.0.json: File that we will use to train our models
- predictions: Folder with contais the predictions (for the questions in dev-v2.0.json)

- main.py: The main code, that executes all
- train.py: The code that trains a model (important: this process take a long time)
===============================
To make the predictions, use the main.py. The output of this must be a json file.
The evaluation (calculate F1-score) is done inside the main.py calling the utils/evaluate-v2.0.py
===============================
Using docker:
docker build -t QAmachine/1.0 .
docker run QAmachine/1.0