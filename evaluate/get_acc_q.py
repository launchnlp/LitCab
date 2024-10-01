from rouge_score import rouge_scorer
import json
import scipy.stats
from sklearn.metrics import roc_auc_score
import sys


# model_gen_file = 'alpaca-nativetest_answers.reeval.semantic_uncertainty.rouge_correctness.bleurt_correctness.jsonl'
# model_gen_file = 'data/llama-7b-hftest_answers.reeval.semantic_uncertainty.rouge_correctness.bleurt_correctness.jsonl'
# model_gen_file = 'data/llama-7b-hftest_answers.reeval.unbias_semantic_uncertainty.rouge_correctness.jsonl'
# model_gen_file = 'data/llama-7b-hftest_answers.reeval_investigate_noise.unbias_semantic_uncertainty.rouge_correctness.jsonl'
# model_gen_file = 'factualityprompt/fever_factual_finalllama-7b-hf_answers.reeval.s_nli_semantic_uncertainty.fact_correctness.jsonl'
# model_gen_file = 'data/test.llama-7b-hf_greedy_search_answers.reeval_investigate_noise.s_nli_semantic_uncertainty.rouge_correctness.jsonl'
# model_gen_file = 'triviaqa/validationllama-7b-hf_answers.reeval.semantic_uncertainty.rouge_correctness.jsonl'
# model_gen_file = 'data/llama-7b-hftest_answers.reeval_investigate_noise.s_nli_unbias_semantic_uncertainty.jsonl'
# model_gen_file = 'factualityprompt/fever_factual_finalllama-7b-hf_answers.reeval_investigate_noise.semantic_uncertainty.fact_correctness.jsonl'
model_gen_file = sys.argv[1]

model_gen = []

with open(model_gen_file, 'r') as f:
    for line in f:
        model_gen.append(json.loads(line))

first_factor = sys.argv[2]
# first_factor = 'avg_entropy_of_first_answer'
# first_factor = 'score_of_biggest_cluster'
# first_factor = 'mean_abs_diff_of_first_answer'
# first_factor = 'mean_abs_diff_of_biggest_cluster'
# first_factor = 'normalized_score'
# first_factor = 'score_of_first_answer'
# first_factor = 'semantic_entropy'
# first_factor = 'proportion_of_biggest_clusters'
# first_factor = "mean_abs_diffs_of_biggest_cluster"
# first_factor = "std_of_mean_abs_diffs_of_biggest_cluster"
# second_factor = 'bleurt_of_biggest_cluster'
# second_factor = 'bleurt_of_first_answer'
# second_factor = 'rougeL_of_biggest_cluster'
# second_factor = 'rougeL_first_answer'
# second_factor = "rouge_correct_first_answer"
# second_factor = "rouge_correct_of_biggest_cluster"
# second_factor = "correct_of_biggest_cluster"
# second_factor = "correct_of_first_answer"
second_factor = sys.argv[3]

q = float(sys.argv[4])
p = float(sys.argv[5])

if first_factor == 'lex_sim&score_of_first_answer':
    first_factor_values = [gen['lex_sim'] + gen['score_of_first_answer'] for gen in model_gen]
elif first_factor == 'score_of_first_answer&QG_score_of_first_answer':
    first_factor_values = [gen['score_of_first_answer'] + gen['QG_score_of_first_answer'] for gen in model_gen]
else:
    if 'entropy' in first_factor or 'num_clusters' in first_factor or "diff" in first_factor or 'std' in first_factor or 'var' in first_factor:
        first_factor_values = [-gen[first_factor] for gen in model_gen]
    else:
        first_factor_values = [gen[first_factor] for gen in model_gen]
second_factor_values = [1 if gen[second_factor] else 0 for gen in model_gen]

# sort first factor values and second factor values by first factor values in descending order
first_factor_values, second_factor_values = zip(*sorted(zip(first_factor_values, second_factor_values), reverse=False))
# print(first_factor_values[:])
# print(second_factor_values)
# print(int(len(second_factor_values) * q))


# correct_scores = [s for s, c in zip(first_factor_values, second_factor_values) if c == 1]
# wrong_scores = [s for s, c in zip(first_factor_values, second_factor_values) if c == 0]
# total_scores = correct_scores + wrong_scores
# median_score = sorted(total_scores)[len(total_scores)//2]
# acc_50 = len([s for s in correct_scores if s >= median_score])/(len(total_scores)//2)
# print(acc_50)

acc_50 = sum(second_factor_values[-int(len(second_factor_values) * q):]) / len(second_factor_values[-int(len(second_factor_values) * q):])
num_correct = 0
cov_09 = 0
for i in range(len(first_factor_values)-1, 0, -1):
    if second_factor_values[i] == 1:
        num_correct += 1
    acc_now = num_correct / (len(first_factor_values) - i)
    if acc_now > p:
        cov_09 = (len(first_factor_values) - i)/len(first_factor_values)

print("acc@q: ", acc_50)
print("cov@p: ", cov_09)

# print(scipy.stats.pearsonr(first_factor_values, second_factor_values)[0])