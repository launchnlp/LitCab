import json
import sys
import pandas as pd

model_gen_file_1000 = sys.argv[1]
output_file = model_gen_file_1000.split('.reeval')[0] + '_1000.csv'
num_samples = 1000

model_gen = []
with open(model_gen_file_1000, 'r') as f:
    for line in f:
        model_gen.append(json.loads(line))

questions = []
answers = []
for i in range(len(model_gen)):
    questions.append(model_gen[i]['question'])
    answers.append(model_gen[i]['answer'])
questions = questions[:num_samples]
answers = answers[:num_samples]

df = pd.DataFrame({'question': questions, 'answer': answers})
df.to_csv(output_file, index=False)