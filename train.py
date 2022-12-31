## This code was based on: https://towardsdatascience.com/how-to-train-bert-for-q-a-in-any-language-63b62c780014

from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import BertTokenizerFast
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AdamW
import torch

## Get the dataset squad_v2
data = load_dataset('squad_v2')

## Function to add the END index in the answers
def add_end_index(answers, contexts):
    new_answers = []
    # loop through each answer-context pair
    for answer, context in tqdm(zip(answers, contexts)):
        ## Check if the answer_start is not null
        if answer['answer_start'] == []:
            answer['answer_start'] = 0
            answer['answer_end'] = 0
            pass
        else:
            # quick reformating to remove lists
            answer['text'] = answer['text'][0]
            answer['answer_start'] = answer['answer_start'][0]
            # gold_text refers to the answer we are expecting to find in context
            gold_text = answer['text']
            # we already know the start index
            start_idx = answer['answer_start']
            # and ideally this would be the end index...
            end_idx = start_idx + len(gold_text)
            # ...however, sometimes squad answers are off by a character or two
            if context[start_idx:end_idx] == gold_text:
                # if the answer is not off :)
                answer['answer_end'] = end_idx
            else:
                # this means the answer is off by 1-2 tokens
                for n in [1, 2]:
                    if context[start_idx-n:end_idx-n] == gold_text:
                        answer['answer_start'] = start_idx - n
                        answer['answer_end'] = end_idx - n
        new_answers.append(answer)
    return new_answers

## Def Split the data and call the function to put the end_answer 
def prep_data(dataset):
    questions = dataset['question']
    contexts = dataset['context']
    answers = add_end_index(
        dataset['answers'],
        contexts
    )
    return {
        'question': questions,
        'context': contexts,
        'answers': answers
    }

dataset = prep_data(data['train'])

## Load the pre-trained model for the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# tokenize
train = tokenizer(dataset['context'], dataset['question'],
                  truncation=True, padding='max_length',
                  max_length=512, return_tensors='pt')
## Load the model
model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')

def add_token_positions(encodings, answers):
    # initialize lists to contain the token indices of answer start/end
    start_positions = []
    end_positions = []
    for i in tqdm(range(len(answers))):
        
        # append start/end token position using char_to_token method
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        # Verify if the answer exist
        if answers[i]['answer_start'] == answers[i]['answer_end']:
            start_positions[-1] = tokenizer.model_max_length
            end_positions[-1] = tokenizer.model_max_length
        else:
            # if start position is None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            # end position cannot be found, char_to_token found space, so shift position until found

            shift = 1
            while end_positions[-1] is None:
                end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
                shift += 1
    # update our encodings object with the new token-based start/end positions
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

# apply function to our data
add_token_positions(train, dataset['answers'])

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# build datasets for both our training data
train_dataset = SquadDataset(train)

loader = torch.utils.data.DataLoader(train_dataset,
                                     batch_size=64,
                                     shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()
optim = AdamW(model.parameters(), lr=1e-4)

for epoch in range(2):
    loop = tqdm(loader)
    for batch in loop:
        optim.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions)
        
        loss = outputs[0]
        loss.backward()
        optim.step()

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

model.save_pretrained('./bert-qa-trained')
