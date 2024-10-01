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

load_from_model = False

# class Calibrator(torch.nn.Module):
#     def __init__(self, hidden_size, vocab_size):
#         super(Calibrator, self).__init__()
#         self.linear = torch.nn.Linear(hidden_size, 200, bias=False)
#         self.linear2 = torch.nn.Linear(200, vocab_size, bias=False)
#         # self.cal_logits = nn.Parameter(torch.zeros(1, vocab_size, device=device))

#     def weight_init(self):
#         nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
#         nn.init.normal_(self.linear2.weight, mean=0.0, std=0.01)
#         # nn.init.zeros_(self.cal_logits)
#         pass

#     def forward(self, hidden_states):
#         return self.linear2(F.relu(self.linear(hidden_states)))
#         # print(hidden_states.shape)
#         # print(self.cal_logits.shape)
#         # print(self.cal_logits.unsqueeze(0).repeat(hidden_states.shape[0], hidden_states.shape[1], 1).shape)
#         # return self.cal_logits.unsqueeze(0).repeat(hidden_states.shape[0], hidden_states.shape[1], 1)

class Calibrator(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(Calibrator, self).__init__()
        self.linear = torch.nn.Linear(hidden_size, vocab_size, bias=True)

    def weight_init(self):
        nn.init.zeros_(self.linear.weight)
        # nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states):
        # return F.sigmoid(self.linear(F.relu(hidden_states)))
        return self.linear(F.relu(hidden_states))

init_temp = 1.0
calibrator = Calibrator(4096,32000).cuda()
calibrator.weight_init()
if load_from_model:
    print("Loading from model")
    p = pipeline('text-generation', model='meta-llama/Llama-2-7b-hf', device=0)
    model = p.model
    lm_head_weight = model.lm_head.weight
    calibrator.linear.weight.data = lm_head_weight.data
    print("Loaded")
    

optimizer = torch.optim.AdamW(calibrator.parameters(), lr=0.000005)
# optimizer = torch.optim.AdamW(calibrator.parameters(), lr=0.00001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
save_interval = 51

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
batch_size = 128
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
        # print(correct_ids)
        wrong_hidden_states = train_c_w_hidden_states[idx][1]
        wrong_logits = train_c_w_logits[idx][1]
        wrong_ids = train_c_w_ids[idx][1]
        # print(wrong_ids)
        # print(correct_ids[0].shape)
        # print(wrong_ids[0].shape)
        # exit()
        batched_correct_hidden_states.extend(correct_hidden_states)
        batched_correct_logits.extend(correct_logits)
        batched_correct_ids.extend(correct_ids)
        batched_wrong_hidden_states.extend(wrong_hidden_states)
        batched_wrong_logits.extend(wrong_logits)
        batched_wrong_ids.extend(wrong_ids)
        # print(len(batched_correct_hidden_states))
        if len(batched_correct_hidden_states) >= batch_size:
            cur_idx += 1
            print(cur_idx)
            batched_correct_hidden_states = torch.cat(batched_correct_hidden_states, dim=1).to(device)
            batched_correct_logits = torch.cat(batched_correct_logits, dim=1).to(device)
            batched_correct_ids = torch.cat(batched_correct_ids, dim=1).to(device)
            # print(batched_correct_ids[0].shape)
            batched_wrong_hidden_states = torch.cat(batched_wrong_hidden_states, dim=1).to(device)
            batched_wrong_logits = torch.cat(batched_wrong_logits, dim=1).to(device)
            batched_wrong_ids = torch.cat(batched_wrong_ids, dim=1).to(device)
            # print(batched_wrong_ids[[0]].shape)
            # exit()
            optimizer.zero_grad()
            correct_bias_logit = calibrator(batched_correct_hidden_states)
            # correct_bias_logit = correct_bias_logit / torch.norm(correct_bias_logit, dim=-1).unsqueeze(-1) * torch.norm(batched_correct_logits, dim=-1).unsqueeze(-1)
            # print(batched_correct_hidden_states.shape)
            # print(batched_correct_logits.shape)
            # print(correct_bias_logit.shape)

            if not load_from_model:
                correct_logits = batched_correct_logits + 1 * correct_bias_logit
                wrong_bias_logit = calibrator(batched_wrong_hidden_states)
                # wrong_bias_logit = wrong_bias_logit / torch.norm(wrong_bias_logit, dim=-1).unsqueeze(-1) * torch.norm(batched_wrong_logits, dim=-1).unsqueeze(-1)
                wrong_logits = batched_wrong_logits + 1 * wrong_bias_logit
            else:
                correct_logits = correct_bias_logit
                wrong_bias_logit = calibrator(batched_wrong_hidden_states)
                wrong_logits = wrong_bias_logit
            # print(correct_logits.shape, batched_correct_ids.shape)
            # exit()
            correct_loss = F.cross_entropy(correct_logits.view(-1, 32000), batched_correct_ids.view(-1), label_smoothing=0.0)
            # wrong_loss = F.cross_entropy(wrong_logits.view(-1, 32000), batched_wrong_ids.view(-1), label_smoothing=0.0).unsqueeze(0)
            # # set a threshold 3 for the wrong loss
            # new_loss = torch.cat([wrong_loss, torch.ones_like(wrong_loss)*3], dim=0)
            # wrong_loss = torch.min(new_loss)
            wrong_loss = F.cross_entropy(wrong_logits.view(-1, 32000), batched_wrong_ids.view(-1), label_smoothing=0.0)
            if wrong_loss.item() < 3.0:
                loss = 1 * correct_loss - wrong_loss + 0.01 * torch.norm(calibrator.linear.weight, p=2) + 3
            else:
                loss = 1 * correct_loss + 0.01 * torch.norm(calibrator.linear.weight, p=2) + 3
            # loss = correct_loss + wrong_loss + 10
            loss.backward()
            optimizer.step()
            # scheduler.step()
            batched_correct_hidden_states = []
            batched_correct_logits = []
            batched_correct_ids = []
            batched_wrong_hidden_states = []
            batched_wrong_logits = []
            batched_wrong_ids = []
            # print(cur_idx)
            # print(cur_idx % eval_interval==0)
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
                            if not load_from_model:
                                correct_logits = batched_correct_logits + 1 * correct_bias_logit
                                wrong_bias_logit = calibrator(batched_wrong_hidden_states)
                                # wrong_bias_logit = wrong_bias_logit / torch.norm(wrong_bias_logit, dim=-1).unsqueeze(-1) * torch.norm(batched_wrong_logits, dim=-1).unsqueeze(-1)
                                wrong_logits = batched_wrong_logits + 1 * wrong_bias_logit
                            else:
                                correct_logits = correct_bias_logit
                                wrong_bias_logit = calibrator(batched_wrong_hidden_states)
                                wrong_logits = wrong_bias_logit
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
        # print(batched_correct_ids[0])
        batched_wrong_hidden_states = torch.cat(batched_wrong_hidden_states, dim=1).to(device)
        batched_wrong_logits = torch.cat(batched_wrong_logits, dim=1).to(device)
        batched_wrong_ids = torch.cat(batched_wrong_ids, dim=1).to(device)
        # print(batched_wrong_ids[[0]])
        # exit()
        optimizer.zero_grad()
        correct_bias_logit = calibrator(batched_correct_hidden_states)
        # correct_bias_logit = correct_bias_logit / torch.norm(correct_bias_logit, dim=-1).unsqueeze(-1) * torch.norm(batched_correct_logits, dim=-1).unsqueeze(-1)
        if not load_from_model:
            correct_logits = batched_correct_logits + 1 * correct_bias_logit
            wrong_bias_logit = calibrator(batched_wrong_hidden_states)
            # wrong_bias_logit = wrong_bias_logit / torch.norm(wrong_bias_logit, dim=-1).unsqueeze(-1) * torch.norm(batched_wrong_logits, dim=-1).unsqueeze(-1)
            wrong_logits = batched_wrong_logits + 1 * wrong_bias_logit
        else:
            correct_logits = correct_bias_logit
            wrong_bias_logit = calibrator(batched_wrong_hidden_states)
            wrong_logits = wrong_bias_logit
        # print(correct_logits.shape, batched_correct_ids.shape)
        # exit()
        correct_loss = F.cross_entropy(correct_logits.view(-1, 32000), batched_correct_ids.view(-1), label_smoothing=0.0)
        # wrong_loss = F.cross_entropy(wrong_logits.view(-1, 32000), batched_wrong_ids.view(-1), label_smoothing=0.0).unsqueeze(0)
        # # set a threshold 3 for the wrong loss
        # new_loss = torch.cat([wrong_loss, torch.ones_like(wrong_loss)*3], dim=0)
        # wrong_loss = torch.min(new_loss)
        wrong_loss = F.cross_entropy(wrong_logits.view(-1, 32000), batched_wrong_ids.view(-1), label_smoothing=0.0)
        if wrong_loss.item() < 3.0:
            loss = 1 * correct_loss - wrong_loss + 0.01 * torch.norm(calibrator.linear.weight, p=2) + 3
        else:
            loss = 1 * correct_loss + 0.01 * torch.norm(calibrator.linear.weight, p=2) + 3
        # loss = correct_loss + wrong_loss + 10
        loss.backward()
        optimizer.step()