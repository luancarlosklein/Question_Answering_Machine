from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import json
import os
import time

def QA(model_name):
    ## Define the pretrained model that will be used
    pretrained_model = model_name
    print(f"Loading: {pretrained_model}")
    ## Define the tokenizer, which depends the model selected
    tokenizer = AutoTokenizer.from_pretrained("models/"+pretrained_model)

    ## Define the model
    model = AutoModelForQuestionAnswering.from_pretrained("models/"+pretrained_model)
    print(f"Loading finished: {pretrained_model}")
    ## Function that recive a question and a text
    ## Return the anwser for the question based on the context
    def answer_question(question, context):
        ## Explanation: https://huggingface.co/docs/transformers/v4.25.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode_plus
        ## Question: Text to be tokenized
        ## Text: Second text to be tokenized
        ## add_special_tolens: Add tokens, special for each model
        ## return_tensors: Equal pt = Return the tensors for the pytorch
        ## truncation: If the context is bigger than the maximum accepted by the model, truncated the text
        inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt", truncation=True)

        ## Pass the inputs to the model
        r = model(**inputs)
        ## Get the start and the end (possibles) of the answer 
        answer_start_scores = r['start_logits']
        answer_end_scores = r['end_logits']

        # Get the most likely beginning of answer with the argmax of the score
        answer_start = torch.argmax(answer_start_scores) 
        answer_end = torch.argmax(answer_end_scores) + 1

        ## Convert the the anwser tokens to a text
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

        ## If the answer is <s> (for Roberta) and [CLS] for (for Bert and deberta), means that there is no answer, so change to ''
        if "<s>" == answer or "[CLS]" == answer:
            answer = ""

        ## Sometimes, the anwser contains the question on the the answer. So, to solve this, remove the question and the special markers
        if answer[0:5] == "[CLS]":
            aux = answer.split("[SEP]")
            answer = aux[1]
        elif answer[0:3] == "<s>":
            aux = answer.split("</s>")
            answer = aux[2]
            
        ## Return the answer
        return answer

    ## Open the file with the questions
    f = open(os.path.join("data", "dev-v2.0.json") )
    data_test = json.load(f)

    ## Create a dict to save the answers
    answers = {}
    cont = 0

    ## Do loop on all the questions
    for i in data_test['data']:
        for j in i['paragraphs']:
            for k in j['qas']:
                ## Get the question
                question = k['question']
                ## Get the question id
                id_q = k['id']
                ## Call the function that answer the question
                answers[id_q] = answer_question(question, j['context'])

                ## Print the id and the question
                cont += 1
                if cont % 100 == 0:
                    print(f"Questions answered: {cont}")

    ## Save the file in a json with all the answers
    with open(os.path.join("predictions", "preds_" + model_name +".json"), "w") as outfile: 
        json.dump(answers, outfile) 

    ## return the quantity of questions
    return cont

## Define de models
models = ["deberta-v3-large-squad2"
          #,"roberta-base-squad2"
          #,"bert-base-cased-squad2"
          ]

## Variable to save the avg time
execution_times = []

## Generate the anwsers
for i in models:
    start = time.time()
    qtd_questions = QA(i)
    end = time.time()
    execution_times.append( (end - start)/qtd_questions)
    
## Calculate the scores
for i in models:
    pred = os.path.join("predictions", "preds_" + i + ".json")
    os.system('python utils/evaluate-v2.0.py data/dev-v2.0.json ' + pred)
