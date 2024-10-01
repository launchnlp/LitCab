import pandas as pd

# file_path = "/../../ICLR_2024/wikiqa/train.txt"
file_path = "/../../ICLR_2024/name_bio/demos.txt"
items = [item for item in open(file_path).read().split('\n\n') if len(item.split('\n')) > 1]
questions = []
answers = []
for item in items:
    if len(item.split('\n')) > 1:
        questions.append(item.split('\n')[0])
        answers.append(item.split('\n')[1])
print(sum([len(a.split()) for a in answers])/len(answers))
# print('\n'.join([str(len(a.split())) for a in answers]))
print(len(answers))
print(max([len(a.split()) for a in answers]))
print(min([len(a.split()) for a in answers]))

# for a in answers:
#     if len(a) > 10:
#         print(a)