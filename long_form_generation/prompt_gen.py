from transformers import pipeline
import pandas as pd
import random
import torch
import nltk
import sys
import json

model_name = sys.argv[1]

prompt_file = "../factuality_prompt/fever_factual_final.jsonl"
if len(sys.argv) > 2 and sys.argv[2] == 'train':
    prompt_file = "../factuality_prompt/train_fever_factual_final.jsonl"


prompts_info = [json.loads(line) for line in open(prompt_file, 'r')]

prompts = [item['prompt'] for item in prompts_info]
topics = [item['evidence_info'] for item in prompts_info]
topics = [', '.join([t[0] for t in e]) for e in topics]
num_prompts = 500

if '13b' in model_name:
    m = pipeline("text-generation", model=model_name, trust_remote_code=True, device=0, torch_dtype=torch.float16)
elif '30b' in model_name or '70b' in model_name:
    m = pipeline("text-generation", model=model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
else:
    m = pipeline("text-generation", model=model_name, trust_remote_code=True, device=0)

demonstration = True
demonstration_file = "../factuality_prompt/demos.txt"
prompt="""SYSTEM: You are an AI research assistant. You use a tone that is technical and scientific.
USER: Hello, who are you?
ASSISTANT: Greeting! I am an AI research assistant. How can I help you today?
USER: """
num_demos=5
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

answers = []
prompts_for_writing = []
topics_for_writing = []
tokenizer = m.tokenizer
stop_id = tokenizer.encode('\n')[-1]

for p, topic in zip(prompts, topics):
    if p == '':
        continue
    print("-------------------")
    print(topic)
    is_repeated = True
    user_prompt = f"Write a paragraph about {topic.split(',')[0]}."
    now_prompt = prompt + user_prompt + f"\nASSISTANT: {topic.split(',')[0]}"
    while(is_repeated):
        gen_answer = m(now_prompt, max_new_tokens=500, do_sample=True, eos_token_id=stop_id)[0]
        answer = gen_answer['generated_text'][len(prompt + user_prompt + "\nASSISTANT: "):]
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
    print(answer)
    answers.append(answer)
    if len(answers) == num_prompts:
        break

df = pd.DataFrame({'prompt': prompts_for_writing, 'topic': topics_for_writing, 'answer': answers})
df.to_csv(prompt_file.replace('.jsonl', f".{model_name.split('/')[-1]}.csv"), index=False)