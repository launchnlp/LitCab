import os
import openai
import pandas as pd
import nltk
import json
from tqdm import tqdm
import time
from transformers import pipeline
import torch
import torch.nn.functional as F
import sys
import random

def longestCommonSubstring(A, B):
    # Initialize the matrix with 0s
    matrix = [[0 for i in range(len(B) + 1)] for j in range(len(A) + 1)]
    # Initialize the maximum length of the substring
    max_length = 0
    # Initialize the index of the maximum length
    index = 0
    # Loop through the matrix
    for i in range(len(A) + 1):
        for j in range(len(B) + 1):
            # If the characters match
            if i > 0 and j > 0 and A[i - 1] == B[j - 1]:
                # Add 1 to the previous diagonal value
                matrix[i][j] = matrix[i - 1][j - 1] + 1
                # If the current length is greater than the maximum length
                if matrix[i][j] > max_length:
                    # Update the maximum length
                    max_length = matrix[i][j]
                    # Update the index
                    index = i
    # Return the substring
    return A[index - max_length:index]

# get the index of a subsequences in a list
def get_subseq_index(subseq, seq):
    subseq_len = len(subseq)
    for i in range(len(seq) - subseq_len + 1):
        if seq[i:i+subseq_len] == subseq:
            return i
    return -1

file_path = sys.argv[1]
model_name = sys.argv[2]
with open(file_path, 'r') as f:
    data = json.load(f)

if '13b' in model_name:
    m = pipeline('text-generation', model=model_name, device=0, torch_dtype=torch.float16)
else:
    m = pipeline('text-generation', model=model_name, device=0)
    
tokenizer = m.tokenizer
model = m.model
prompt="""SYSTEM: SYSTEM: You are an AI research assistant. You use a tone that is technical and scientific.
USER: Hello, who are you?
ASSISTANT: Greeting! I am an AI research assistant. How can I help you today?
USER: """

names = data.keys()
store_hidden_states_file = "../name_bio" + "/" + model_name.split('/')[-1] + "correct_wrong_hidden_states.pt"
store_logits_file = "../name_bio" + "/" + model_name.split('/')[-1] + "_correct_wrong_logits.pt"
store_ids_file = "../name_bio" + "/" + model_name.split('/')[-1] + "_correct_wrong_ids.pt"

demonstration_file = "../name_bio/demos.txt"
num_demos=5
demonstration=True
if demonstration:
    demo_questions = [item.split('\n')[0] for item in open(demonstration_file).read().split('\n\n') if len(item.split('\n')) > 1]
    demo_answers = [item.split('\n')[1] for item in open(demonstration_file).read().split('\n\n') if len(item.split('\n')) > 1]
    # randomly select num_demos questions and corrisponding answers from the demonstration file
    qa_pairs = list(zip(demo_questions, demo_answers))
    random.shuffle(qa_pairs)
    qa_pairs = qa_pairs[:num_demos]
    # random.shuffle(qa_pairs)
    for qa_pair in qa_pairs:
        prompt += qa_pair[0] + '\nASSISTANT: ' + qa_pair[1] + '\nUSER: '

total_hidden_states = []
total_logits = []
total_ids = []

for name in tqdm(names):
    bio = data[name]['bio']
    if "cannot provide a bio" in bio:
        continue
    facts = data[name]['facts']
    segs = data[name]['segs']
    correctness = data[name]['facts_correctness']
    
    LCS_segs = [longestCommonSubstring(bio, seg) for seg in segs]

    question = f"Write a paragraph for {name}'s biography."
    now_prompt = prompt + question + f'\nASSISTANT: ' + bio
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoding = tokenizer(now_prompt.strip(), return_tensors='pt').to(device)
    # print(encoding)
    # print(encoding['input_ids'].data.cpu().tolist()[0])
    # exit()
    seg_scores = []
    with torch.no_grad():
        encoding['labels'] = encoding['input_ids'].clone()
        encoding['output_hidden_states'] = True
        outputs = model(**encoding)

        logits = outputs.logits
        # print(logits.shape)
        # print(hidden_states.shape)
        # exit()
        logits = logits[..., :-1, :].contiguous()
        encoding['labels'] = encoding['labels'][..., 1:].contiguous()
        hidden_states = outputs.hidden_states[-1]
        hidden_states = hidden_states[..., :-1, :].contiguous()

        whole_prompt_list = encoding['input_ids'].data.cpu().tolist()[0]
        correct_hidden_states = []
        correct_logits = []
        correct_ids = []
        wrong_hidden_states = []
        wrong_logits = []
        wrong_ids = []
        for LCS_seg, c in zip(LCS_segs, correctness):
            LCS_seg_encoding = tokenizer(LCS_seg, return_tensors='pt').to(device)
            LCS_seg_list = LCS_seg_encoding['input_ids'].data.cpu().tolist()[0]
            LCS_seg_list = longestCommonSubstring(whole_prompt_list, LCS_seg_list)
            # print(whole_prompt_list)
            # print(LCS_seg_list)

            # print(LCS_seg)
            # print(bio)
            LCS_seg_index = get_subseq_index(LCS_seg_list, whole_prompt_list)

            seg_logits = logits[:, LCS_seg_index : LCS_seg_index+len(LCS_seg_list)]
            seg_hidden_states = hidden_states[:, LCS_seg_index : LCS_seg_index+len(LCS_seg_list)]
            scores_of_label = [s.item() for s in F.softmax(seg_logits / 1, dim=-1).gather(2, encoding['labels'][:, LCS_seg_index : LCS_seg_index+len(LCS_seg_list)].unsqueeze(-1)).squeeze(-1)[0]]
            normalized_score = torch.exp(sum([torch.log(torch.tensor(s)) for s in scores_of_label])/len(scores_of_label)).item()

            logit_write = seg_logits.data.cpu()
            hidden_states_write = seg_hidden_states.data.cpu()
            ids_write = encoding['labels'][:, LCS_seg_index : LCS_seg_index+len(LCS_seg_list)].data.cpu()
            # normalized_score = sum(scores_of_label)/len(scores_of_label)
            # normalized_score = min(scores_of_label)
            # print(scores_of_label)
            # print(normalized_score)
            # exit()
            seg_scores.append(normalized_score)
            if c:
                correct_hidden_states.append(hidden_states_write)
                correct_logits.append(logit_write)
                correct_ids.append(ids_write)
            else:
                wrong_hidden_states.append(hidden_states_write)
                wrong_logits.append(logit_write)
                wrong_ids.append(ids_write)

        total_hidden_states.append([correct_hidden_states, wrong_hidden_states])
        total_logits.append([correct_logits, wrong_logits])
        total_ids.append([correct_ids, wrong_ids])
    
torch.save(total_hidden_states, store_hidden_states_file)
torch.save(total_logits, store_logits_file)
torch.save(total_ids, store_ids_file)