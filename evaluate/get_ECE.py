import json
import sys


# model_gen_file = 'alpaca-nativetest_answers.reeval.semantic_uncertainty.rouge_correctness.bleurt_correctness.jsonl'
# model_gen_file = 'data/llama-7b-hftest_answers.reeval.semantic_uncertainty.rouge_correctness.bleurt_correctness.jsonl'
# model_gen_file = 'data/llama-7b-hftest_answers.reeval.unbias_semantic_uncertainty.rouge_correctness.jsonl'
# model_gen_file = 'data/llama-7b-hftest_answers.reeval_investigate_noise.unbias_semantic_uncertainty.rouge_correctness.jsonl'
# model_gen_file = 'factualityprompt/fever_factual_finalllama-7b-hf_answers.reeval.s_nli_semantic_uncertainty.fact_correctness.jsonl'
# model_gen_file = 'data/llama-7b-hftest_answers.reeval_investigate_noise.s_nli_unbias_semantic_uncertainty_by_rouge.rouge_correctness.jsonl'
# model_gen_file = 'factualityprompt/fever_factual_finalllama-7b-hf_answers.reeval.unbias_semantic_uncertainty.fact_correctness.jsonl'
# model_gen_file = 'data/test.llama-7b-hf_greedy_search_answers.reeval_investigate_noise.s_nli_semantic_uncertainty.rouge_correctness.jsonl'
# model_gen_file = 'factualityprompt/fever_factual_finalllama-7b-hf_answers.reeval.semantic_uncertainty.fact_correctness.jsonl'
model_gen_file = sys.argv[1]

model_gen = []

with open(model_gen_file, 'r') as f:
    for line in f:
        model_gen.append(json.loads(line))

# first_factor = 'avg_entropy_of_first_answer'
# first_factor = 'proportion_of_biggest_clusters'
# first_factor = 'score_of_biggest_cluster'
# first_factor = 'mean_abs_diff_of_first_answer'
# first_factor = 'mean_abs_diff_of_biggest_cluster'
# first_factor = 'normalized_score'
# first_factor = 'score_of_first_answer'
# first_factor = 'semantic_entropy'
# first_factor = 'num_clusters'
first_factor = sys.argv[2]
# second_factor = 'bleurt_of_biggest_cluster'
# second_factor = 'bleurt_of_first_answer'
# second_factor = 'rougeL_of_biggest_cluster'
# second_factor = 'rougeL_first_answer'
# second_factor = "rouge_correct_first_answer"
# second_factor = "rouge_correct_of_biggest_cluster"
# second_factor = "correct_of_biggest_cluster"
# second_factor = "correct_of_first_answer"
second_factor = sys.argv[3]
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

min_first_factor = min(first_factor_values)
max_first_factor = max(first_factor_values)

# # group second values into 10 bins according to first factor values
# bins = [[] for _ in range(10)]
# for i in range(len(second_factor_values)):
#     first_factor_value = first_factor_values[i]
#     second_factor_value = second_factor_values[i]
#     bin_index = int((first_factor_value - min_first_factor-0.000001) / (max_first_factor - min_first_factor) * 10)
#     bins[bin_index].append(second_factor_value)

# # group second values into 10 bins according to first factor values, each bin has the same number of samples
# num_samples_per_bin = len(second_factor_values) // 10
# sorted_indices = sorted(range(len(first_factor_values)), key=lambda k: first_factor_values[k])
# bins = [[] for _ in range(10)]
# for i in range(10):
#     bin_start = i * num_samples_per_bin
#     bin_end = (i+1) * num_samples_per_bin
#     if i == 9:
#         bin_end = len(first_factor_values)
#     bin_indices = sorted_indices[bin_start:bin_end]
#     for index in bin_indices:
#         bins[i].append(second_factor_values[index])

# # calculate mean of each bin
# means = []
# for i, bin in enumerate(bins):
#     if len(bin) == 0:
#         means.append(0.0)
#         # print(0.0)
#     else:
#         means.append(sum(bin) / len(bin))
#         # print(sum(bin) / len(bin))

# # calculate ECE
# ece = 0
# for i, bin in enumerate(bins):
#     ece += abs(means[i] - (i+1)/10) * len(bin) / len(second_factor_values)
# # print("ECE: ", ece)
# print(ece)
# # for i, bin in enumerate(bins):
#     # print(len(bin))

# if any first factor value is less than 0, scale all first factor values to be 0 to 1
if first_factor == "lex_sim" or first_factor == "semantic_entropy":
    first_factor_values = [value - min_first_factor for value in first_factor_values]
    max_first_factor = max(first_factor_values)
    first_factor_values = [value / max_first_factor for value in first_factor_values]

first_factor_values = [value - min_first_factor for value in first_factor_values]
max_first_factor = max(first_factor_values)
first_factor_values = [value / max_first_factor for value in first_factor_values]

bin_size = 0.1
bins = [0.0]
while bins[-1] < 1.0:
    bins.append(bins[-1] + bin_size)
bins[-1] = 1.0
bin2score = {}
for bin_idx in range(len(bins)-1):
    bin2score[bin_idx] = []
for score, label in zip(first_factor_values, second_factor_values):
    bin_idx = int(score/bin_size)
    # print(score)
    # print(bin_idx)
    bin2score[bin_idx].append(label)
bin2acc = {}
for bin_idx in bin2score:
    bin2acc[bin_idx] = sum(bin2score[bin_idx])/len(bin2score[bin_idx]) if len(bin2score[bin_idx]) > 0 else 0
    print(bin2acc[bin_idx])
ece = 0
for bin_idx in bin2acc:
    ece += abs(bin2acc[bin_idx] - (bins[bin_idx] + bins[bin_idx+1])/2) * len(bin2score[bin_idx])
ece /= len(first_factor_values)
print('ece: ', ece)

for bin_idx in bin2acc:
    print(len(bin2score[bin_idx]))