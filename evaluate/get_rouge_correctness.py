from rouge_score import rouge_scorer
import json
import numpy as np
import sys

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# model_gen_file = 'alpaca-nativetest_answers.reeval.semantic_uncertainty.jsonl'
# model_gen_file = 'data/llama-7b-hftest_answers.reeval_investigate_noise.s_nli_unbias_semantic_uncertainty_by_rouge.jsonl'
# model_gen_file = 'data/testllama-7b-hf_greedy_search_answers.reeval_investigate_noise.semantic_uncertainty.jsonl'
# model_gen_file = 'triviaqa/validationllama-7b-hf_answers.reeval.semantic_uncertainty.jsonl'
# model_gen_file = 'data/testllama-7b-hftemperature_1.0topp_0.5_answers.reeval.semantic_uncertainty.jsonl'
model_gen_file = sys.argv[1]

model_gen = []
# answer_file = "data/test.txt"
# answer_file = 'triviaqa/validation.txt'
answer_file = sys.argv[2]

with open(model_gen_file, 'r') as f:
    for line in f:
        model_gen.append(json.loads(line))

q2a = {}
with open(answer_file, 'r') as f:
    answers = [item for item in f.read().split('\n\n') if item != '']
    for answer in answers:
        qa = answer.split('\n')
        q = qa[0]
        a = qa[1]
        q2a[q] = a

out_file = model_gen_file.replace('.jsonl', '.rouge_correctness.jsonl')

total_len = 0
count = 0

with open(out_file, 'w') as f:
    for gen in model_gen:
        question = gen['question']
        gen_answers = gen['answer']
        clusters = gen['clusters']
        len_clusters = [len(cluster) for cluster in clusters]
        max_cluster = clusters[len_clusters.index(max(len_clusters))]
        gen['proportion_of_biggest_clusters'] = len(max_cluster) / len(gen_answers)
        ans_idx = max_cluster[0] # get the first answer in the largest cluster
        gen_answer = gen_answers[ans_idx]

        total_len += len(gen['scores'][0])
        count += 1

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
        gen['avg_score'] = sum(gen['normalized_score']) / len(gen['normalized_score'])
        gen['D_var'] = np.std(gen['normalized_score'])
        rougeL_first_answer = scorer.score(q2a[question], gen_answers[0])
        gen['rougeL_first_answer'] = rougeL_first_answer['rougeL'].fmeasure
        if rougeL_first_answer['rougeL'].fmeasure > 0.3:
            gen['rouge_correct_first_answer'] = True
        else:
            gen['rouge_correct_first_answer'] = False

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
            # gen['mean_abs_diffs_of_first_answer'] = mean_abs_diffs_of_first_answer[0]
            # gen['mean_abs_diffs_of_biggest_cluster'] = mean_abs_diffs_of_biggest_cluster[0]
            gen['std_of_mean_abs_diffs_of_first_answer'] = np.std(mean_abs_diffs_of_first_answer)
            gen['std_of_mean_abs_diffs_of_biggest_cluster'] = np.std(mean_abs_diffs_of_biggest_cluster)
        
        sim_scores = []
        for r in range(len(gen_answers)):
            for c in range(len(gen_answers)):
                score = scorer.score(gen_answers[r], gen_answers[c])
                sim_scores.append(score['rougeL'].fmeasure)
        gen['lex_sim'] = sum(sim_scores) / len(sim_scores)

        f.write(json.dumps(gen) + '\n')
        f.flush()

print(total_len / count)