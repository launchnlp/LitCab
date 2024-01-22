import os
import openai
import pandas as pd
import nltk
import json
from tqdm import tqdm
import time
import sys

openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") 
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

facts_user_content = """Please break down the following sentence into independent facts. You should ONLY present the independent facts (one in a row), no other words or explaination.
-> """
# segs_user_content = """Which segment in the following sentence reflects the fact “THIS IS THE FACT”? The segment doesn't need to be a complete sentence and should be as short as possible.
# -> """
segs_user_content = """Which part of the following paragraph reflects the fact “THIS IS THE FACT”? The part doesn't need to be a complete sentence and should be as short as possible.
"""


model_name = sys.argv[1]
if len(sys.argv) > 2 and sys.argv[2] == 'train':
    bio_path = f"../factuality_prompt/train_fever_factual_final.{model_name.split('/')[-1]}.csv"
    bio_pd = pd.read_csv(bio_path)
    output_path = f"../factuality_prompt/train_fever_factual_final.{model_name.split('/')[-1]}.bio_facts_segs.json"
else:
    bio_path = f"../factuality_prompt/fever_factual_final.{model_name.split('/')[-1]}.csv"
    bio_pd = pd.read_csv(bio_path)
    output_path = f"../factuality_prompt/fever_factual_final.{model_name.split('/')[-1]}.bio_facts_segs.json"
name2bio = {}
for i in range(len(bio_pd)):
    name = bio_pd.iloc[i]['topic']
    bio = bio_pd.iloc[i]['answer']
    name2bio[name] = bio

name2info = {}

count = 0

for name in tqdm(name2bio):
    if count >= 100:
        break
    count += 1
    print("*************************************")
    print(name)
    # print(name2bio[name])
    if "personal information" in name2bio[name]:
        continue
    sents = name2bio[name].split('\n')
    total_facts = []
    total_segs = []
    for sent in sents:
        if sent == '':
            continue
        print("-------------------------------------")
        print(sent)
        now_user_content = facts_user_content + sent
        try:
            response = openai.ChatCompletion.create(
                engine="gpt35", # engine = "deployment_name".
                # engine="gpt4",
                messages=[
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": now_user_content}
                ]
            )
            res = response['choices'][0]['message']['content']
        except:
            print(response)
            continue
        facts = [s.replace('- ', '') for s in res.split('\n')]
        segs = []
        for fact in facts:
            now_seg_content = segs_user_content.replace("THIS IS THE FACT", fact) + sent
            try:
                response = openai.ChatCompletion.create(
                    engine="gpt35", # engine = "deployment_name".
                    # engine="gpt4",
                    messages=[
                        {"role": "user", "content": now_seg_content}
                    ]
                )
                now_seg = response['choices'][0]['message']['content']
            except:
                print(response)
                continue
            segs.append(now_seg)
            time.sleep(0.1)
        total_facts.extend(facts)
        total_segs.extend(segs)
    name2info[name] = {}
    name2info[name]['bio'] = name2bio[name]
    name2info[name]['facts'] = total_facts
    name2info[name]['segs'] = total_segs
    # break
    time.sleep(0.5)

with open(output_path, 'w') as f:
    json.dump(name2info, f, indent=4)
