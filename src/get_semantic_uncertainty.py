from transformers import pipeline
import pandas as pd
import json
import math
from tqdm import tqdm
import sys

nli_model = pipeline("text-classification", model="microsoft/deberta-large-mnli", device=0)

model_gen_file = sys.argv[1]

bi_nli = True
normalize_score = True
unbias = False

model_gen = []

with open(model_gen_file, 'r') as f:
    for line in f:
        model_gen.append(json.loads(line))

if bi_nli:
    out_file = model_gen_file.replace('.jsonl', '.semantic_uncertainty.jsonl')
else:
    out_file = model_gen_file.replace('.jsonl', '.s_nli_semantic_uncertainty.jsonl')
if unbias:
    out_file = out_file.replace('.jsonl', '.unbias_semantic_uncertainty.jsonl')

with open(out_file, 'w') as f:
    for i in tqdm(range(len(model_gen))):
        if 'factualityprompt' in model_gen_file:
            question = model_gen[i]['prompt']
            answers = model_gen[i]['continuation']
        else:
            question = model_gen[i]['question']
            answers = model_gen[i]['answer']
        adj_mat = [[0 for _ in range(len(answers))] for _ in range(len(answers))]

        prompts = []
        for r in range(len(answers)):
            for c in range(len(answers)):
                p = {}
                p['text'] = answers[r]
                p['text_pair'] = answers[c]
                prompts.append(p)
        
        results = nli_model(prompts, batch_size=100)
        res_labels = [result['label'] for result in results]

        for r in range(len(answers)):
            for c in range(len(answers)):
                if res_labels[r*len(answers) + c] == 'ENTAILMENT':
                    adj_mat[r][c] = 1
        
        if bi_nli:
            bi_adj_mat = [[0 for _ in range(len(answers))] for _ in range(len(answers))]
            for r in range(len(answers)):
                for c in range(len(answers)):
                    if adj_mat[r][c] == 1 and adj_mat[c][r] == 1:
                        bi_adj_mat[r][c] = 1
            adj_mat = bi_adj_mat

        clusters = [[idx] for idx in range(len(answers))]
        for r in range(len(answers)):
            for c in range(len(answers)):
                if adj_mat[r][c] == 1:
                    if r != c:
                        r_index = -1
                        c_index = -1
                        for idx in range(len(clusters)):
                            if r in clusters[idx]:
                                r_index = idx
                            if c in clusters[idx]:
                                c_index = idx
                        if r_index != c_index:
                            clusters[r_index] = clusters[r_index] + clusters[c_index]
                            clusters[c_index] = []
        clusters = [cluster for cluster in clusters if len(cluster) > 0]

        model_gen[i]['clusters'] = clusters
        if normalize_score:
            scores = model_gen[i]['scores']
            scores = [[s if s!=0 else 1e-10 for s in score] for score in scores]
            scores_of_each_ans = [math.exp(sum([math.log(s) for s in score])) for score in scores]
        else:
            scores = model_gen[i]['scores']
            scores_of_each_ans = [math.exp(sum([math.log(s) for s in score])) for score in scores]
        # normalize scores
        sum_scores = sum(scores_of_each_ans)
        scores_of_each_ans = [score / sum_scores for score in scores_of_each_ans]
        if len(scores_of_each_ans) != len(answers):
            print('scores and answers do not match')
            print(question)
            print(answers)
            exit()
        scores_of_each_cluster = []
        for cluster in clusters:
            scores_of_each_cluster.append(sum([scores_of_each_ans[idx] for idx in cluster]))
        model_gen[i]['scores_of_each_cluster'] = scores_of_each_cluster
        if not unbias:
            entropy = -sum([p * math.log2(p) / len(scores_of_each_cluster) for p in scores_of_each_cluster])
        else:
            entropy = -sum([1/len(scores_of_each_cluster) * math.log2(p) for p in scores_of_each_cluster])
        model_gen[i]['semantic_entropy'] = entropy
        model_gen[i]['num_clusters'] = len(clusters)
        len_clusters = [len(cluster) for cluster in clusters]
        max_cluster = clusters[len_clusters.index(max(len_clusters))]
        model_gen[i]['proportion_of_biggest_clusters'] = len(max_cluster) / len(answers)

        f.write(json.dumps(model_gen[i]) + '\n')
        f.flush()