import json
from nltk import sent_tokenize
from transformers import pipeline
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
from torch import nn
import pandas as pd7
import sys
import pandas as pd

seed = 42
random.seed(seed)

# model_name = "decapoda-research/llama-7b-hf"
# model_name = "meta-llama/Llama-2-7b-hf"
model_name = sys.argv[1]
dataset_name = sys.argv[2]
p_n_path = "../" + dataset_name + "/train.pos_neg.3_strong_negs.csv"

# question_file = "../sciq/test.txt"
demonstration = True
# demonstration_file = "../sciq/train.txt"
demonstration_file = "../" + dataset_name + "/train.txt"
store_hidden_states_file = "../" + dataset_name + "/" + model_name.split('/')[-1] + "_correct_wrong_hidden_states.pt"
store_logits_file = "../" + dataset_name + "/" + model_name.split('/')[-1] + "_correct_wrong_logits.pt"
store_ids_file = "../" + dataset_name + "/" + model_name.split('/')[-1] + "_correct_wrong_ids.pt"
device = 0
num_demos = 15
# do_sample = True if num_return_sequences > 1 else False
do_sample = True
temperature = 1
top_p=1.0
total_num_questions = 1000

if '13b' in model_name:
    generator = pipeline("text-generation", model=model_name, trust_remote_code=True, device=device, torch_dtype=torch.float16)
elif '30b' in model_name or '70b' in model_name:
    generator = pipeline("text-generation", model=model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
else:
    generator = pipeline("text-generation", model=model_name, trust_remote_code=True, device=device)


model = generator.model
tokenizer = generator.tokenizer

correct_wrong_qa_info_raw = pd.read_csv(p_n_path)
correct_wrong_qa_info = []

for i in range(len(correct_wrong_qa_info_raw)):
    correct_wrong_qa_info.append({'question': correct_wrong_qa_info_raw['question'][i], 'correct_answer': correct_wrong_qa_info_raw['correct_answer'][i] if type(correct_wrong_qa_info_raw['correct_answer'][i])==str else eval(correct_wrong_qa_info_raw['correct_answer'][i]), 'wrong_answer': [sent_tokenize(s)[0] if not s.startswith('1.') else s for s in eval(correct_wrong_qa_info_raw['wrong_answer'][i]) if s != '']})

prompt = """SYSTEM: You are an AI research assistant. You use a tone that is technical and scientific.
USER: Hello, who are you?
ASSISTANT: Greeting! I am an AI research assistant. How can I help you today?
USER: """

if demonstration:
    demo_questions = [item.split('\n')[0] for item in open(demonstration_file).read().split('\n\n') if len(item.split('\n')) > 1]
    demo_answers = [item.split('\n')[1] for item in open(demonstration_file).read().split('\n\n') if len(item.split('\n')) > 1]
    # randomly select num_demos questions and corrisponding answers from the demonstration file
    qa_pairs = list(zip(demo_questions, demo_answers))
    random.shuffle(qa_pairs)
    qa_pairs = qa_pairs[:num_demos]
    for qa_pair in qa_pairs:
        prompt += qa_pair[0] + '\nASSISTANT: ' + qa_pair[1] + '\nUSER: '
# print(prompt)
# exit()
class Calibrator(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(Calibrator, self).__init__()
        self.linear = torch.nn.Linear(hidden_size, vocab_size)

    def weight_init(self):
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states):
        return self.linear(hidden_states)

init_temp = 1.0
calibrator = Calibrator(4096,32000).cuda()
calibrator.weight_init()
optimizer = torch.optim.Adam(calibrator.parameters(), lr=0.00001)
update_interval = 50
save_interval = 500
total_hidden_states = []
total_logits = []
total_ids = []

for idx, c_w_info in tqdm(enumerate(correct_wrong_qa_info)):
    # try:
    q = c_w_info['question']
    correct_q = q
    wrong_q = q
    if type(c_w_info['correct_answer']) == str:
        correct_as = [c_w_info['correct_answer']]
    else:
        correct_as = c_w_info['correct_answer']
    wrong_as = c_w_info['wrong_answer']
    # print("-------------------")
    # print(q)

    correct_hidden_states = []
    correct_logits = []
    correct_ids = []
    for i in range(len(correct_as)):
        correct_a = correct_as[i]
        now_correct_prompt = prompt + correct_q + '\nASSISTANT: ' + correct_a
        with torch.no_grad():
            encoding = tokenizer(now_correct_prompt.strip(), return_tensors='pt').to(generator.device)
            encoding['labels'] = encoding['input_ids'].clone()
            encoding['output_hidden_states'] = True
            outputs = model(**encoding)

            # compute the loss of new tokens
            logits = outputs.logits
            hidden_states = outputs.hidden_states

            hidden_states_write = hidden_states[-1][..., :-1, :].contiguous()
            without_answer = prompt + correct_q + '\nASSISTANT: '
            hidden_states_write = hidden_states_write[:, len(tokenizer.encode(without_answer.strip()))-1:].to('cpu')
            correct_hidden_states.append(hidden_states_write)

            logits_write = logits[..., :-1, :].contiguous()
            logits_write = logits_write[:, len(tokenizer.encode(without_answer.strip()))-1:].to('cpu')
            correct_logits.append(logits_write)

            encoding['labels'] = encoding['labels'][..., 1:].contiguous()
            ids_write = encoding['labels'][:, len(tokenizer.encode(without_answer.strip()))-1:]
            correct_ids.append(ids_write)

        # calibrated_logits = calibrator(hidden_states[-1]) + logits
        # logits = calibrated_logits[..., :-1, :].contiguous()
        # encoding['labels'] = encoding['labels'][..., 1:].contiguous()
        # without_answer = prompt + correct_q + '\nASSISTANT: '
        # new_logits = logits[:, len(tokenizer.encode(without_answer.strip()))-1:]
        # # scores_of_label = [s for s in calibrator(new_logits).gather(2, encoding['labels'][:, len(tokenizer.encode(without_answer.strip()))-1:][0]]
        # # correct_nll = -sum([math.log(s) for s in scores_of_label])
        # correct_nll_loss.append(F.cross_entropy(new_logits.view(-1, new_logits.size(-1)), encoding['labels'][:, len(tokenizer.encode(without_answer.strip()))-1:].view(-1), reduction='mean'))

    wrong_hidden_states = []
    wrong_logits = []
    wrong_ids = []
    for i in range(len(wrong_as)):
        wrong_a = wrong_as[i]
        # compute the loss of wrong tokens
        now_wrong_prompt = prompt + wrong_q + '\nASSISTANT: ' + wrong_a
        # print(now_wrong_prompt)
        with torch.no_grad():
            encoding = tokenizer(now_wrong_prompt.strip(), return_tensors='pt').to(generator.device)
            encoding['labels'] = encoding['input_ids'].clone()
            encoding['output_hidden_states'] = True
            outputs = model(**encoding)

            # compute the loss of new tokens
            logits = outputs.logits
            hidden_states = outputs.hidden_states

            hidden_states_write = hidden_states[-1][..., :-1, :].contiguous()
            without_answer = prompt + wrong_q + '\nASSISTANT: '
            hidden_states_write = hidden_states_write[:, len(tokenizer.encode(without_answer.strip()))-1:].to('cpu')
            wrong_hidden_states.append(hidden_states_write)

            logits_write = logits[..., :-1, :].contiguous()
            logits_write = logits_write[:, len(tokenizer.encode(without_answer.strip()))-1:].to('cpu')
            wrong_logits.append(logits_write)

            encoding['labels'] = encoding['labels'][..., 1:].contiguous()
            ids_write = encoding['labels'][:, len(tokenizer.encode(without_answer.strip()))-1:]
            wrong_ids.append(ids_write)

    cur_c_w_hidden_states = [correct_hidden_states, wrong_hidden_states]
    cur_c_w_logits = [correct_logits, wrong_logits]
    cur_c_w_ids = [correct_ids, wrong_ids]

    total_hidden_states.append(cur_c_w_hidden_states)
    total_logits.append(cur_c_w_logits)
    total_ids.append(cur_c_w_ids)
    # except Exception as e:
    #     print(e)
    #     continue

torch.save(total_hidden_states, store_hidden_states_file)
torch.save(total_logits, store_logits_file)
torch.save(total_ids, store_ids_file)