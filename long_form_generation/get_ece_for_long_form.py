import pandas as pd
import json
import sys

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
    
# classifying the scores into 10 bins
bin_size = 0.1
bins = [0.0]
while bins[-1] < 1.0:
    bins.append(bins[-1] + bin_size)
bins[-1] = 1.0
bin2score = {}
for bin_idx in range(len(bins)-1):
    bin2score[bin_idx] = []
for score, label in zip(all_scores, all_labels):
    bin_idx = int(score/bin_size)
    bin2score[bin_idx].append(label)
bin2acc = {}
for bin_idx in bin2score:
    bin2acc[bin_idx] = sum(bin2score[bin_idx])/len(bin2score[bin_idx]) if len(bin2score[bin_idx]) > 0 else 0
    print(bin2acc[bin_idx])
ece = 0
for bin_idx in bin2acc:
    ece += abs(bin2acc[bin_idx] - (bins[bin_idx] + bins[bin_idx+1])/2) * len(bin2score[bin_idx])
ece /= len(all_scores)
for bin_idx in bin2score:
    print(len(bin2score[bin_idx]))
print('ece: ', ece)