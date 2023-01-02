# si-project
Files:
<br>- utils/evaluate-v2.0.py: Function evaluation already done. It is the official evaluation.
<br>- data/dev-v2.0.json: Our test file. Answering these questions, we obtain the evaluation
<br>- data/train-v2.0.json: File that we will use to train our models
<br>- predictions: Folder with contais the predictions (for the questions in dev-v2.0.json)
<br>- models: Constains the models (The model has a big size. To dowload it, clone inside this filder the follow repository: git clone https://huggingface.co/deepset/deberta-v3-large-squad2)

<br>- main.py: The main code, that executes all
<br>- train.py: The code that trains a model (important: this process take a long time)
<br> --------------------------------------------------------

<br>To make the predictions, use the main.py. The output of this must be a json file.
<br>The evaluation (calculate F1-score) is done inside the main.py calling the utils/evaluate-v2.0.py
<br> --------------------------------------------------------

<br>Using docker:
<br>docker build -t qamachine/1.0 .
<br>docker run PYTHONUNBUFFERED=1 -d qamachine/1.0
