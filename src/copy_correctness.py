import json
import sys
from rouge_score import rouge_scorer


model_gen_file = sys.argv[1]
model_gen_file_with_correctness = sys.argv[2]
output_file = model_gen_file.replace('.jsonl', '.gpt4_correctness.jsonl')

model_gen = []
with open(model_gen_file, 'r') as f:
    for line in f:
        model_gen.append(json.loads(line))

model_gen_with_correctness = []
with open(model_gen_file_with_correctness, 'r') as f:
    for line in f:
        # print(line)
        model_gen_with_correctness.append(json.loads(line))
q2b_correctness = {}
q2f_correctness = {}
for i in range(len(model_gen_with_correctness)):
    q2b_correctness[model_gen_with_correctness[i]['question']] = model_gen_with_correctness[i]['rouge_correct_of_biggest_cluster']
    q2f_correctness[model_gen_with_correctness[i]['question']] = model_gen_with_correctness[i]['rouge_correct_first_answer']

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
for i in range(len(model_gen)):
    model_gen[i]['rouge_correct_of_biggest_cluster'] = q2b_correctness[model_gen[i]['question']]
    model_gen[i]['rouge_correct_first_answer'] = q2f_correctness[model_gen[i]['question']]
    model_gen[i]['score_of_first_answer'] = model_gen[i]['normalized_score'][0]
    gen_answers = model_gen[i]['answer']
    sim_scores = []
    for r in range(len(gen_answers)):
        for c in range(len(gen_answers)):
            score = scorer.score(gen_answers[r], gen_answers[c])
            sim_scores.append(score['rougeL'].fmeasure)
    model_gen[i]['lex_sim'] = sum(sim_scores) / len(sim_scores)

with open(output_file, 'w') as f:
    for i in range(len(model_gen)):
        f.write(json.dumps(model_gen[i]) + '\n')