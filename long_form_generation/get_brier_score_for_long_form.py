from rouge_score import rouge_scorer
import json
import scipy.stats
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
import sys


file_path = sys.argv[1]
lines = [line for line in open(file_path, 'r').read().split('\n') if len(line) > 0]
all_scores = []
all_labels = []
for line in lines:
    data = json.loads(line)
    if len(data['seg_scores']) == len(data['facts_correctness']):
        all_scores += data['seg_scores']
        all_labels += [1 if l else 0 for l in data['facts_correctness']]

first_factor_values = all_scores
second_factor_values = all_labels

brier_score = brier_score_loss(second_factor_values, first_factor_values)
print(brier_score)