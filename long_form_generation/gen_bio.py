from transformers import pipeline
import pandas as pd
import random
import torch
import nltk
import sys

model_name = sys.argv[1]

if '13b' in model_name:
    m = pipeline("text-generation", model=model_name, trust_remote_code=True, device=0, torch_dtype=torch.float16)
elif '30b' in model_name or '70b' in model_name:
    m = pipeline("text-generation", model=model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
else:
    m = pipeline("text-generation", model=model_name, trust_remote_code=True, device=0)

demonstration = True
demonstration_file = "../name_bio/demos.txt"
people_name_file = "../name_bio/prompt_entities.txt"

names = [item.strip() for item in open(people_name_file).read().split('\n')]
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
names_for_writing = []
tokenizer = m.tokenizer
stop_id = tokenizer.encode('\n')[-1]

for name in names:
    if name == '':
        continue
    print("-------------------")
    print(name)
    is_repeated = True
    question = f"Write a paragraph for {name}'s biography."
    now_prompt = prompt + question + f'\nASSISTANT: {name}'
    while(is_repeated):
        gen_answer = m(now_prompt, max_new_tokens=500, do_sample=True, eos_token_id=stop_id)[0]
        answer = gen_answer['generated_text']
        answer = answer.split('ASSISTANT: ')[-1]
        # detect whether there repeated sentences in the answer
        sentences = nltk.sent_tokenize(answer)
        if len(sentences) != len(set(sentences)):
            print("Repeated sentences detected!")
            print(answer)
            print("Re-generating...")
            continue
        is_repeated = False

    names_for_writing.append(name)
    print(answer)
    answers.append(answer)

df = pd.DataFrame({'name': names_for_writing, 'answer': answers})
df.to_csv(people_name_file.replace('.txt', f".{model_name.split('/')[-1]}.csv"), index=False)