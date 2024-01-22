from transformers import pipeline
import pandas as pd
import random
import torch
import nltk
import sys
import json

model_name = sys.argv[1]

prompt_file = "/data/xin/ICLR_2024/factuality_prompt/fever_factual_final.jsonl"
prompts_info = [json.loads(line) for line in open(prompt_file, 'r')]

output_file_path = "/data/xin/ICLR_2024/factuality_prompt/lora_train.txt"

prompts = [item['prompt'] for item in prompts_info][-500:]
topics = [item['evidence_info'] for item in prompts_info][-500:]
topics = [', '.join([t[0] for t in e]) for e in topics]
num_prompts = 500

if '13b' in model_name:
    m = pipeline("text-generation", model=model_name, trust_remote_code=True, device=0, torch_dtype=torch.float16)
elif '30b' in model_name or '70b' in model_name:
    m = pipeline("text-generation", model=model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
else:
    m = pipeline("text-generation", model=model_name, trust_remote_code=True, device=0)

answers = []
prompts_for_writing = []
topics_for_writing = []
tokenizer = m.tokenizer

output_file = open(output_file_path, 'w')

for prompt, topic in zip(prompts, topics):
    if prompt == '':
        continue
    print("-------------------")
    print(prompt)
    is_repeated = True
    now_prompt = prompt
    while(is_repeated):
        gen_answer = m(now_prompt, max_new_tokens=200, do_sample=True)[0]
        answer = gen_answer['generated_text']
        # detect whether there repeated sentences in the answer
        sentences = nltk.sent_tokenize(answer)
        if len(sentences) != len(set(sentences)):
            print("Repeated sentences detected!")
            print(answer)
            print("Re-generating...")
            continue
        is_repeated = False

    prompts_for_writing.append(prompt)
    topics_for_writing.append(topic)
    output_file.write(answer + '\n')
    answers.append(answer)
    if len(answers) == num_prompts:
        break
