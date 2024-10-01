from rouge_score import rouge_scorer
import json
import numpy as np
import sys

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# model_gen_file = 'alpaca-nativetest_answers.reeval.semantic_uncertainty.jsonl'
# model_gen_file = 'data/llama-7b-hftest_answers.reeval_investigate_noise.s_nli_unbias_semantic_uncertainty_by_rouge.jsonl'
# model_gen_file = 'data/testllama-7b-hf_greedy_search_answers.reeval_investigate_noise.semantic_uncertainty.jsonl'
# model_gen_file = 'triviaqa/validationllama-7b-hf_answers.reeval.semantic_uncertainty.jsonl'
# model_gen_file = 'data/testllama-7b-hftemperature_1.0topp_0.5_answers.reeval.semantic_uncertainty.jsonl'
model_gen_file = sys.argv[1]

model_gen = []
# answer_file = "data/test.txt"
# answer_file = 'triviaqa/validation.txt'
answer_file = sys.argv[2]

with open(model_gen_file, 'r') as f:
    for line in f:
        model_gen.append(json.loads(line))

q2a = {}
with open(answer_file, 'r') as f:
    answers = [item for item in f.read().split('\n\n') if item != '']
    for answer in answers:
        qa = answer.split('\n')
        q = qa[0]
        a = qa[1]
        q2a[q] = a

out_file = model_gen_file.replace('.jsonl', '.rougeL.jsonl')

total_len = 0
count = 0

with open(out_file, 'w') as f:
    for gen in model_gen:
        question = gen['question']
        gen_answers = gen['answer']
        clusters = gen['clusters']
        len_clusters = [len(cluster) for cluster in clusters]
        max_cluster = clusters[len_clusters.index(max(len_clusters))]
        gen['proportion_of_biggest_clusters'] = len(max_cluster) / len(gen_answers)
        ans_idx = max_cluster[0] # get the first answer in the largest cluster

        total_len += len(gen['scores'][0])
        count += 1

        # compute rougeL score
        rougeL = []
        for gen_answer in gen_answers:
            rougeL.append(scorer.score(q2a[question], gen_answer)['rougeL'].fmeasure)
            
        gen['rougeL'] = rougeL
        f.write(json.dumps(gen) + '\n')
        f.flush()

print(total_len / count)