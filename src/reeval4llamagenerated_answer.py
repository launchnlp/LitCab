from transformers import pipeline
# from transformers import GPTJForCausalLM, AutoTokenizer, GPT2LMHeadModel, T5ForConditionalGeneration
from tqdm import tqdm
import torch
import pandas as pd
import random
from nltk.tokenize import sent_tokenize
import torch.nn.functional as F
from transformers import (
     LogitsProcessorList,
     MinLengthLogitsProcessor,
     StoppingCriteriaList,
     RepetitionPenaltyLogitsProcessor,
     MaxLengthCriteria,
 )
import json
from tqdm import tqdm
import sys

seed = 42
random.seed(seed)

if_compute_loss = False
demonstration = True
num_demos=15

model_gen_path = sys.argv[1]
model_name = sys.argv[2]
demonstration_file = sys.argv[3]
if len(sys.argv) > 4:
    temperature = float(sys.argv[4])
else:
    temperature = 1

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

model_gens = pd.read_csv(model_gen_path)
model_gen = []

for i in range(len(model_gens)):
    model_gen.append({'question': model_gens['question'][i], 'answer': [sent_tokenize(s)[0] if not s.startswith('1.') else s for s in eval(model_gens['answer'][i]) if s != '']})

if '13b' in model_name:
    p = pipeline('text-generation', model=model_name, device=0, torch_dtype=torch.float16)
elif '30b' in model_name or '70b' in model_name:
    p = pipeline("text-generation", model=model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
else:
    p = pipeline('text-generation', model=model_name, device=0)
tokenizer = p.tokenizer
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token # to avoid an error
model = p.model
model.eval()

if temperature == 1:
    output_file = open(model_gen_path.replace('.csv', '.reeval.jsonl'), 'w')
else:
    output_file = open(model_gen_path.replace('.csv', '.reeval.temperature_' + str(temperature) + '.jsonl'), 'w')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

for idx, (question, anses) in tqdm(enumerate(zip(model_gens['question'].tolist(), model_gens['answer'].tolist()))):

    ans_tokens = []
    scores_of_label_list = []
    normed_scores = []
    entropys = []
    evg_entropys = []
    anses = eval(anses)

    for ans in anses:
        ans = ans.strip()
        now_prompts = prompt + question + '\nASSISTANT: ' + ans

        encoding = tokenizer(now_prompts.strip(), return_tensors='pt').to(device)
        with torch.no_grad():
            encoding['labels'] = encoding['input_ids'].clone()
            outputs = model(**encoding)

            # compute the loss of new tokens
            logits = outputs.logits
            logits = logits[..., :-1, :].contiguous()
            encoding['labels'] = encoding['labels'][..., 1:].contiguous()
            without_answer = prompt + question + '\nASSISTANT: '
            new_logits = logits[:, len(tokenizer.encode(without_answer.strip()))-1:]
            entropy = [e.item() for e in -(F.softmax(new_logits / temperature, dim=-1) * F.log_softmax(new_logits, dim=-1)).sum(-1).squeeze(0)]
            scores_of_label = [s.item() for s in F.softmax(new_logits / temperature, dim=-1).gather(2, encoding['labels'][:, len(tokenizer.encode(without_answer.strip()))-1:].unsqueeze(-1)).squeeze(-1)[0]]

            if len(scores_of_label) == 0:
                print("no scores")
                print(now_prompts)
                print(len(new_logits))
                continue

            normalized_score = torch.exp(sum([torch.log(torch.tensor(s)) for s in scores_of_label])/len(scores_of_label)).item()

            ans_tokens.append(tokenizer.convert_ids_to_tokens(encoding['input_ids'][0][len(tokenizer.encode(without_answer.strip())):]))
            normed_scores.append(normalized_score)
            scores_of_label_list.append(scores_of_label)
            entropys.append(entropy)
            avg_entropy = sum(entropy)/len(entropy)
            evg_entropys.append(avg_entropy)


    model_gen[idx]['scores'] = scores_of_label_list
    model_gen[idx]['normalized_score'] = normed_scores
    model_gen[idx]['tokens'] = ans_tokens
    model_gen[idx]['entropy'] = entropys
    model_gen[idx]['avg_entropy'] = evg_entropys

    output_file.write(json.dumps(model_gen[idx]) + '\n')
    output_file.flush()

print(output_file.name)