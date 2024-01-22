import pandas as pd
import json
import sys
from sklearn.metrics import roc_auc_score

file_path = sys.argv[1]
lines = [line for line in open(file_path, 'r').read().split('\n') if len(line) > 0]
all_scores = []
all_labels = []
for line in lines:
    data = json.loads(line)
    if len(data['seg_scores']) != len(data['facts_correctness']):
        continue
    all_scores += data['seg_scores']
    all_labels += [1 if l else 0 for l in data['facts_correctness']]

auroc = roc_auc_score(all_labels, all_scores)
print(auroc)