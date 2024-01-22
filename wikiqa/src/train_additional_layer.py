import json
from nltk import sent_tokenize
from transformers import pipeline
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
from torch import nn
import pandas as pd7
import sys
import pandas as pd

seed = 42
random.seed(seed)

dataset_name = sys.argv[1]
model_name = sys.argv[2]
hidden_states_path =  "../" + dataset_name + "/" + model_name.split('/')[-1] + "_correct_wrong_hidden_states.pt"
logits_path = "../" + dataset_name + "/" + model_name.split('/')[-1] + "_correct_wrong_logits.pt"
ids_path = "../" + dataset_name + "/" + model_name.split('/')[-1] + "_correct_wrong_ids.pt"
question_file = "../" + dataset_name + "/test.txt"
demonstration_file = "../" + dataset_name + "/train.txt"
device = 0
eval_interval = 5
loss_print_interval = 5


class Calibrator(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(Calibrator, self).__init__()
        self.linear = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def weight_init(self):
        nn.init.zeros_(self.linear.weight)

    def forward(self, hidden_states):
        return self.linear(F.relu(hidden_states))

init_temp = 1.0
calibrator = Calibrator(4096,32000).cuda()
calibrator.weight_init()
optimizer = torch.optim.AdamW(calibrator.parameters(), lr=0.000005)
save_interval = 100

c_w_hidden_states = torch.load(hidden_states_path)
c_w_logits = torch.load(logits_path)
c_w_ids = torch.load(ids_path)

for idx in range(len(c_w_hidden_states)):

    c_w_hidden_states[idx] = (c_w_hidden_states[idx][0][:], c_w_hidden_states[idx][1][:])
    c_w_logits[idx] = (c_w_logits[idx][0][:], c_w_logits[idx][1][:])
    c_w_ids[idx] = (c_w_ids[idx][0][:], c_w_ids[idx][1][:])

# shuffle the data
c_w_hidden_states, c_w_logits, c_w_ids = zip(*random.sample(list(zip(c_w_hidden_states, c_w_logits, c_w_ids)), len(c_w_hidden_states)))

train_c_w_hidden_states = c_w_hidden_states[:int(len(c_w_hidden_states)*0.8)]
train_c_w_logits = c_w_logits[:int(len(c_w_logits)*0.8)]
train_c_w_ids = c_w_ids[:int(len(c_w_ids)*0.8)]
val_c_w_hidden_states = c_w_hidden_states[int(len(c_w_hidden_states)*0.8):]
val_c_w_logits = c_w_logits[int(len(c_w_logits)*0.8):]
val_c_w_ids = c_w_ids[int(len(c_w_ids)*0.8):]

calibrator.train()
batch_size = 64
eval_batch_size = 32

cur_idx = 0
lower_bound_loss = 9999
for cur_epoch in tqdm(range(20)):
    print("Epoch: ", cur_epoch)
    batched_correct_hidden_states = []
    batched_correct_logits = []
    batched_correct_ids = []
    batched_wrong_hidden_states = []
    batched_wrong_logits = []
    batched_wrong_ids = []

    for idx in range(len(train_c_w_hidden_states)):
        correct_hidden_states = train_c_w_hidden_states[idx][0]
        correct_logits = train_c_w_logits[idx][0]
        correct_ids = train_c_w_ids[idx][0]
        wrong_hidden_states = train_c_w_hidden_states[idx][1]
        wrong_logits = train_c_w_logits[idx][1]
        wrong_ids = train_c_w_ids[idx][1]
        batched_correct_hidden_states.extend(correct_hidden_states)
        batched_correct_logits.extend(correct_logits)
        batched_correct_ids.extend(correct_ids)
        batched_wrong_hidden_states.extend(wrong_hidden_states)
        batched_wrong_logits.extend(wrong_logits)
        batched_wrong_ids.extend(wrong_ids)
        if len(batched_correct_hidden_states) >= batch_size:
            cur_idx += 1
            batched_correct_hidden_states = torch.cat(batched_correct_hidden_states, dim=1).to(device)
            batched_correct_logits = torch.cat(batched_correct_logits, dim=1).to(device)
            batched_correct_ids = torch.cat(batched_correct_ids, dim=1).to(device)
            batched_wrong_hidden_states = torch.cat(batched_wrong_hidden_states, dim=1).to(device)
            batched_wrong_logits = torch.cat(batched_wrong_logits, dim=1).to(device)
            batched_wrong_ids = torch.cat(batched_wrong_ids, dim=1).to(device)
            optimizer.zero_grad()
            correct_bias_logit = calibrator(batched_correct_hidden_states)
            correct_logits = batched_correct_logits + 1 * correct_bias_logit
            wrong_bias_logit = calibrator(batched_wrong_hidden_states)
            wrong_logits = batched_wrong_logits + 1 * wrong_bias_logit
            correct_loss = F.cross_entropy(correct_logits.view(-1, 32000), batched_correct_ids.view(-1), label_smoothing=0.0)
            wrong_loss = F.cross_entropy(wrong_logits.view(-1, 32000), batched_wrong_ids.view(-1), label_smoothing=0.0)
            if wrong_loss.item() < 3.0:
                loss = 1 * correct_loss - wrong_loss + 0.01 * torch.norm(calibrator.linear.weight, p=2) + 3
            else:
                loss = 1 * correct_loss + 0.01 * torch.norm(calibrator.linear.weight, p=2) + 3
            loss.backward()
            optimizer.step()
            # scheduler.step()
            batched_correct_hidden_states = []
            batched_correct_logits = []
            batched_correct_ids = []
            batched_wrong_hidden_states = []
            batched_wrong_logits = []
            batched_wrong_ids = []
            if cur_idx % loss_print_interval == 0:
                print("Loss: ", loss)
                print("Correct Loss: ", correct_loss)
                print("Wrong Loss: ", wrong_loss)
            if cur_idx % eval_interval == 0:
                total_eval_loss = 0
                total_correct_loss = 0
                total_wrong_loss = 0
                count = 0 
                with torch.no_grad():
                    for val_idx in range(len(val_c_w_hidden_states)):
                        correct_hidden_states = val_c_w_hidden_states[val_idx][0]
                        correct_logits = val_c_w_logits[val_idx][0]
                        correct_ids = val_c_w_ids[val_idx][0]
                        wrong_hidden_states = val_c_w_hidden_states[val_idx][1]
                        wrong_logits = val_c_w_logits[val_idx][1]
                        wrong_ids = val_c_w_ids[val_idx][1]
                        batched_correct_hidden_states.extend(correct_hidden_states)
                        batched_correct_logits.extend(correct_logits)
                        batched_correct_ids.extend(correct_ids)
                        batched_wrong_hidden_states.extend(wrong_hidden_states)
                        batched_wrong_logits.extend(wrong_logits)
                        batched_wrong_ids.extend(wrong_ids)
                        if len(batched_correct_hidden_states) == eval_batch_size:
                            batched_correct_hidden_states = torch.cat(batched_correct_hidden_states, dim=1).to(device)
                            batched_correct_logits = torch.cat(batched_correct_logits, dim=1).to(device)
                            batched_correct_ids = torch.cat(batched_correct_ids, dim=1).to(device)
                            batched_wrong_hidden_states = torch.cat(batched_wrong_hidden_states, dim=1).to(device)
                            batched_wrong_logits = torch.cat(batched_wrong_logits, dim=1).to(device)
                            batched_wrong_ids = torch.cat(batched_wrong_ids, dim=1).to(device)
                            correct_bias_logit = calibrator(batched_correct_hidden_states)
                            # correct_bias_logit = correct_bias_logit / torch.norm(correct_bias_logit, dim=-1).unsqueeze(-1) * torch.norm(batched_correct_logits, dim=-1).unsqueeze(-1)
                            correct_logits = batched_correct_logits + 1 * correct_bias_logit
                            wrong_bias_logit = calibrator(batched_wrong_hidden_states)
                            # wrong_bias_logit = wrong_bias_logit / torch.norm(wrong_bias_logit, dim=-1).unsqueeze(-1) * torch.norm(batched_wrong_logits, dim=-1).unsqueeze(-1)
                            wrong_logits = batched_wrong_logits + 1 * wrong_bias_logit
                            correct_loss = F.cross_entropy(correct_logits.view(-1, 32000), batched_correct_ids.view(-1), label_smoothing=0.0)
                            wrong_loss = F.cross_entropy(wrong_logits.view(-1, 32000), batched_wrong_ids.view(-1), label_smoothing=0.0)
                            loss = 1 * correct_loss - wrong_loss + 3
                            batched_correct_hidden_states = []
                            batched_correct_logits = []
                            batched_correct_ids = []
                            batched_wrong_hidden_states = []
                            batched_wrong_logits = []
                            batched_wrong_ids = []
                            total_eval_loss += loss.item()
                            total_correct_loss += correct_loss.item()
                            total_wrong_loss += wrong_loss.item()
                            count += 1
                    if len(batched_correct_hidden_states) > 0:
                        batched_correct_hidden_states = torch.cat(batched_correct_hidden_states, dim=1).to(device)
                        batched_correct_logits = torch.cat(batched_correct_logits, dim=1).to(device)
                        batched_correct_ids = torch.cat(batched_correct_ids, dim=1).to(device)
                        batched_wrong_hidden_states = torch.cat(batched_wrong_hidden_states, dim=1).to(device)
                        batched_wrong_logits = torch.cat(batched_wrong_logits, dim=1).to(device)
                        batched_wrong_ids = torch.cat(batched_wrong_ids, dim=1).to(device)
                        correct_bias_logit = calibrator(batched_correct_hidden_states)
                        # correct_bias_logit = correct_bias_logit / torch.norm(correct_bias_logit, dim=-1).unsqueeze(-1) * torch.norm(batched_correct_logits, dim=-1).unsqueeze(-1)
                        correct_logits = batched_correct_logits + 1 * correct_bias_logit
                        wrong_bias_logit = calibrator(batched_wrong_hidden_states)
                        # wrong_bias_logit = wrong_bias_logit / torch.norm(wrong_bias_logit, dim=-1).unsqueeze(-1) * torch.norm(batched_wrong_logits, dim=-1).unsqueeze(-1)
                        wrong_logits = batched_wrong_logits + 1 * wrong_bias_logit
                        correct_loss = F.cross_entropy(correct_logits.view(-1, 32000), batched_correct_ids.view(-1), label_smoothing=0.0)
                        wrong_loss = F.cross_entropy(wrong_logits.view(-1, 32000), batched_wrong_ids.view(-1), label_smoothing=0.0)
                        loss = 1 * correct_loss - wrong_loss + 3
                        # print(correct_logits)
                        # print(batched_correct_ids)
                        batched_correct_hidden_states = []
                        batched_correct_logits = []
                        batched_correct_ids = []
                        batched_wrong_hidden_states = []
                        batched_wrong_logits = []
                        batched_wrong_ids = []
                        total_eval_loss += loss.item()
                        total_correct_loss += correct_loss.item()
                        total_wrong_loss += wrong_loss.item()
                        count += 1
                    print("Val correct Loss: ", total_correct_loss/count)
                    print("Val wrong Loss: ", total_wrong_loss/count)
                    print("Val Loss: ", total_eval_loss/count)
                    # exit()
                    if total_eval_loss/count < lower_bound_loss:
                        lower_bound_loss = total_eval_loss/count
                        torch.save(calibrator.state_dict(), "../" + dataset_name + "/calibrator/" + model_name.split('/')[-1] + "_calibrator_best.pt")
            if cur_idx % save_interval == 0:
                print("Saving model: ", cur_idx)
                torch.save(calibrator.state_dict(), "../" + dataset_name + "/calibrator/" + model_name.split('/')[-1] + "_calibrator_" + str(cur_idx) + ".pt")
    if len(batched_correct_hidden_states) > 0:
        cur_idx += 1
        batched_correct_hidden_states = torch.cat(batched_correct_hidden_states, dim=1).to(device)
        batched_correct_logits = torch.cat(batched_correct_logits, dim=1).to(device)
        batched_correct_ids = torch.cat(batched_correct_ids, dim=1).to(device)
        batched_wrong_hidden_states = torch.cat(batched_wrong_hidden_states, dim=1).to(device)
        batched_wrong_logits = torch.cat(batched_wrong_logits, dim=1).to(device)
        batched_wrong_ids = torch.cat(batched_wrong_ids, dim=1).to(device)
        optimizer.zero_grad()
        correct_bias_logit = calibrator(batched_correct_hidden_states)
        correct_logits = batched_correct_logits + 1 * correct_bias_logit
        wrong_bias_logit = calibrator(batched_wrong_hidden_states)
        wrong_logits = batched_wrong_logits + 1 * wrong_bias_logit
        correct_loss = F.cross_entropy(correct_logits.view(-1, 32000), batched_correct_ids.view(-1), label_smoothing=0.0)
        wrong_loss = F.cross_entropy(wrong_logits.view(-1, 32000), batched_wrong_ids.view(-1), label_smoothing=0.0)
        if wrong_loss.item() < 3.0:
            loss = 1 * correct_loss - wrong_loss + 0.01 * torch.norm(calibrator.linear.weight, p=2) + 3
        else:
            loss = 1 * correct_loss + 0.01 * torch.norm(calibrator.linear.weight, p=2) + 3
        loss.backward()
        optimizer.step()