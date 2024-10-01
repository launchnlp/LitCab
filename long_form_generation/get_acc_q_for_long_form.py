from rouge_score import rouge_scorer
import json
import scipy.stats
from sklearn.metrics import roc_auc_score
import sys
import nltk

file_path = sys.argv[1]
lines = [line for line in open(file_path, 'r').read().split('\n') if len(line) > 0]
all_scores = []
all_labels = []

answers = []
total_len_ans = 0
count = 0
min_len = 9999999
max_len = 0
for line in lines:
    data = json.loads(line)
    if len(data['seg_scores']) == len(data['facts_correctness']):
        all_scores += data['seg_scores']
        all_labels += [1 if l else 0 for l in data['facts_correctness']]
        total_len_ans += len(nltk.word_tokenize(data['bio']))
        count += 1
        if len(nltk.word_tokenize(data['bio'])) > max_len:
            max_len = len(nltk.word_tokenize(data['bio']))
        if len(nltk.word_tokenize(data['bio'])) < min_len:
            min_len = len(nltk.word_tokenize(data['bio']))
print("avg len of bio: ", total_len_ans/count)
print("max len of bio: ", max_len)
print("min len of bio: ", min_len)
q = float(sys.argv[2])
p = float(sys.argv[3])
first_factor_values = all_scores
second_factor_values = all_labels

# sort first factor values and second factor values by first factor values
first_factor_values, second_factor_values = zip(*sorted(zip(first_factor_values, second_factor_values)))
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
