# si-project
Files:
<br>- utils/evaluate-v2.0.py: Function evaluation already done. It is the official evaluation.
<br>- data/dev-v2.0.json: Our test file. Answering these questions, we obtain the evaluation
<br>- data/train-v2.0.json: Training dataset
<br>- predictions: Folder with contais the predictions (for the questions in dev-v2.0.json)
<br>- models: Constains the models (The model has a big size. To download it, clone inside this filder the follow repository: git clone https://huggingface.co/deepset/deberta-v3-large-squad2)

<br>- main.py: The main code, that executes all
<br>- train.py: The code that trains a model (important: this process take a long time)
<br> --------------------------------------------------------

<br>To generate all the predictions, the main.py is executed. The output of this must be a json file.
<br>The evaluation (calculate F1-score) is also done inside the main.py calling the utils/evaluate-v2.0.py
<br> --------------------------------------------------------

<br>Dockerfiles:
<br>The first dockerfile named "Dockerfile" executes just the evaluation script with the predictions already generated.
<br>docker build -t qamachine:1.0 .
<br>docker run PYTHONUNBUFFERED=1 -d qamachine:1.0
<br>The second dockerfile named "Dockerfile-generate" executes the main code, which generates all the predictions, and the evaluation in sequence.
<br>docker build -f Dockerfile-generate -t qamachine-generate:1.0 .
<br>docker run PYTHONUNBUFFERED=1 -d qamachine-generate:1.0
