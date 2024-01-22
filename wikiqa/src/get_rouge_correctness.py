from rouge_score import rouge_scorer
import json
import numpy as np
import sys

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

model_gen_file = sys.argv[1]
answer_file = sys.argv[2]

model_gen = []

with open(model_gen_file, 'r') as f:
    for line in f:
        model_gen.append(json.loads(line))

q2a = {}
with open(answer_file, 'r') as f:
    answers = [item for item in f.read().split('\n\n') if item != '']
    for answer in answers:
        qa = answer.split('\n')
        q = qa[0]
        if q[-1] == '?' and q[-2] != ' ':
            q = q[:-1]
        if q[-1] == '?' and q[-2] == ' ':
            q = q[:-2]
        a = qa[1]
        q2a[q] = a

out_file = model_gen_file.replace('.jsonl', '.rouge_correctness.jsonl')

total_len = 0
count = 0
total_rougeL = 0
total_acc = 0

words_total_len = 0
answer_words_total_len = 0

with open(out_file, 'w') as f:
    for gen in model_gen:
        if "verbalization_prob" in gen:
            if gen['verbalization_prob'] == "<the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>":
                continue
            gen['verbalization_prob'] = float(gen['verbalization_prob'])
        question = gen['question'] if gen['question'][0] != ' ' else gen['question'][1:]
        if question[-1] == '?' and question[-2] == ' ':
            question = question[:-2]
        if question[-1] == '?' and question[-2] != ' ':
            question = question[:-1]
        if "Which 'Inn of Court' in London is missing - Lincoln's Inn, Inner Temple, Middle Temple and" in question:
            question = "Which 'Inn of Court' in London is missing - Lincoln's Inn, Inner Temple, Middle Temple and ........"
        gen_answers = gen['answer']
        clusters = gen['clusters']
        len_clusters = [len(cluster) for cluster in clusters]
        max_cluster = clusters[len_clusters.index(max(len_clusters))]
        gen['proportion_of_biggest_clusters'] = len(max_cluster) / len(gen_answers)
        ans_idx = max_cluster[0] # get the first answer in the largest cluster
        gen_answer = gen_answers[ans_idx]

        total_len += len(gen['scores'][0])
        count += 1
        words_total_len += len(gen_answer.split())
        answer_words_total_len += len(q2a[question].split())

        # compute rougeL score
        rougeL = scorer.score(q2a[question], gen_answer)
        gen['rougeL_of_biggest_cluster'] = rougeL['rougeL'].fmeasure
        if rougeL['rougeL'].fmeasure > 0.3:
            gen['rouge_correct_of_biggest_cluster'] = True
        else:
            gen['rouge_correct_of_biggest_cluster'] = False
        gen['gold_answer'] = q2a[question]
        gen['score_of_biggest_cluster'] = gen['normalized_score'][ans_idx]
        gen['score_of_first_answer'] = gen['normalized_score'][0]
        gen['max_score_of_first_answer'] = max(gen['scores'][0])
        gen['min_score_of_first_answer'] = min(gen['scores'][0])
        gen['avg_score'] = sum(gen['normalized_score']) / len(gen['normalized_score'])
        gen['D_var'] = np.std(gen['normalized_score'])
        rougeL_first_answer = scorer.score(q2a[question], gen_answers[0])
        total_rougeL += rougeL_first_answer['rougeL'].fmeasure
        gen['rougeL_first_answer'] = rougeL_first_answer['rougeL'].fmeasure
        if rougeL_first_answer['rougeL'].fmeasure > 0.3:
            gen['rouge_correct_first_answer'] = True
        else:
            gen['rouge_correct_first_answer'] = False
        total_acc += 1 if gen['rouge_correct_first_answer'] else 0

        scores_of_biggest_cluster = gen['scores'][ans_idx]
        scores_of_first_answer = gen['scores'][0]
        gen['std_of_biggest_cluster'] = np.std(scores_of_biggest_cluster)
        gen['std_of_first_answer'] = np.std(scores_of_first_answer)

        avg_entropys = []
        for entropy in gen['entropy']:
            avg_entropys.append(sum(entropy)/len(entropy))
        gen['avg_entropy'] = avg_entropys
        gen['avg_entropy_of_first_answer'] = avg_entropys[0]

        if "mean_abs_diffs_for_q" in gen:
            mean_abs_diffs_of_first_answer = gen['mean_abs_diffs_for_q'][0]
            mean_abs_diffs_of_biggest_cluster = gen['mean_abs_diffs_for_q'][ans_idx]
            gen['mean_abs_diffs_of_first_answer'] = sum(mean_abs_diffs_of_first_answer) / len(mean_abs_diffs_of_first_answer)
            gen['mean_abs_diffs_of_biggest_cluster'] = sum(mean_abs_diffs_of_biggest_cluster) / len(mean_abs_diffs_of_biggest_cluster)
            gen['std_of_mean_abs_diffs_of_first_answer'] = np.std(mean_abs_diffs_of_first_answer)
            gen['std_of_mean_abs_diffs_of_biggest_cluster'] = np.std(mean_abs_diffs_of_biggest_cluster)
        
        sim_scores = []
        for r in range(len(gen_answers)):
            for c in range(len(gen_answers)):
                score = scorer.score(gen_answers[r], gen_answers[c])
                sim_scores.append(score['rougeL'].fmeasure)
        gen['lex_sim'] = sum(sim_scores) / len(sim_scores)

        if "QG_scores" in gen:
            gen['QG_score_of_biggest_cluster'] = gen['QG_normalized_score'][ans_idx]
            gen['QG_score_of_first_answer'] = gen['QG_normalized_score'][0]
            gen['avg_QG_score'] = sum(gen['QG_normalized_score']) / len(gen['QG_normalized_score'])
            gen['D_var_QG'] = np.std(gen['QG_normalized_score'])
            gen['std_of_QG_scores_of_biggest_cluster'] = np.std(gen['QG_scores'][ans_idx])
            gen['std_of_QG_scores_of_first_answer'] = np.std(gen['QG_scores'][0])
            avg_QG_entropy = []
            for entropy in gen['QG_entropy']:
                avg_QG_entropy.append(sum(entropy)/len(entropy))
            gen['avg_QG_entropy'] = avg_QG_entropy
            gen['avg_QG_entropy_of_first_answer'] = avg_QG_entropy[0]
        
        if "normalized_scores_of_each_ensemble" in gen:
            gen['std_normalized_scores_of_each_ensemble'] = np.std(gen['normalized_scores_of_each_ensemble'])
        

        f.write(json.dumps(gen) + '\n')
        f.flush()

print('avg words len')
print(words_total_len / count)
print('avg answer words len')
print(answer_words_total_len / count)
print('avg sent len')
print(total_len / count)
print('avg rougel')
print(total_rougeL / count)
print('avg acc')
print(total_acc / count)