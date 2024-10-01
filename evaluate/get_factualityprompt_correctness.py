import json
import sys
sys.path.append('../../FactualityPrompt')
sys.path.append('..')
from factualityprompt.retriever import obtain_relevant_evidences, get_wiki_from_db
from src.claim_handling import obtain_important_ne, has_incorrect_style
from src.factuality_metric import nli_metric, ner_metric, nli_metric_batch
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
# model_gen_file = 'alpaca-nativetest_answers.reeval.semantic_uncertainty.jsonl'
# model_gen_file = 'factualityprompt/fever_factual_finalllama-7b-hf_answers.reeval_investigate_noise.semantic_uncertainty.jsonl'
model_gen_file = sys.argv[1]
model_gen = []
answer_file = '../factualityprompt/fever_factual_final.jsonl'

with open(model_gen_file, 'r') as f:
    for line in f:
        model_gen.append(json.loads(line))

p2e = {}
with open(answer_file, 'r') as f:
    lines = f.read().split('\n')
    for line in lines:
        if line:
            data = json.loads(line)
            p2e[data['prompt']] = data['evidence_info']

def single_eval(gen_answer):
    gen_sent_ne = obtain_important_ne(gen_answer.strip())
    claim_to_verify = gen_sent_ne['gen']
    evs  = obtain_relevant_evidences(claim_to_verify, wiki_sentences, k=1, method='combined') # will return 2 x k sentences
    # print('claim_to_verify: ', claim_to_verify)
    # print('evidence: ', evs)


    NE_to_check = gen_sent_ne['important_ne'] + gen_sent_ne['unimportant_ne']
    if len(NE_to_check) == 0:
        return 0, 0
    # print('NE_to_check: ', NE_to_check)

    correct_ner_ratio = ner_metric(NE_to_check, wiki_sentences) # apply directly on wiki
    
    hallu_ner_ratio = 1 - correct_ner_ratio


    premise_hypothesis_pairs = [[ev[0], claim_to_verify] for ev in evs]
    nli_probs, labels = nli_metric_batch(premise_hypothesis_pairs)

    entailment_argmax = np.argmax([nli_s[2] for nli_s in nli_probs])
    max_prob = nli_probs[entailment_argmax]
    max_label = labels[entailment_argmax]
    used_ev = evs[entailment_argmax]


    # print(max_prob, max_label)
    nli_contradict_prob = max_prob[0] 
    nli_neutral_prob = max_prob[1] 
    nli_entail_prob = max_prob[2]

    nli_label = max_label

    return hallu_ner_ratio, nli_label


out_file = model_gen_file.replace('.jsonl', '.fact_correctness.jsonl')
with open(out_file, 'w') as f:
    for gen in tqdm(model_gen):
        prompt = gen['prompt']
        gen_answers = gen['continuation']
        evidence_info = [e[0] for e in p2e[prompt]]
        print(prompt)
        wiki_sentences = get_wiki_from_db(evidence_info)
        clusters = gen['clusters']
        len_clusters = [len(cluster) for cluster in clusters]
        max_cluster = clusters[len_clusters.index(max(len_clusters))]
        ans_idx = max_cluster[0] # get the first answer in the largest cluster

        gen['proportion_of_biggest_clusters'] = len(max_cluster) / len(gen_answers)
        gen['score_of_biggest_cluster'] = gen['normalized_score'][ans_idx]
        gen['score_of_first_answer'] = gen['normalized_score'][0]
        gen['avg_score'] = sum(gen['normalized_score']) / len(gen['normalized_score'])
        gen['D_var'] = np.std(gen['normalized_score'])

        avg_entropys = []
        for entropy in gen['entropy']:
            avg_entropys.append(sum(entropy)/len(entropy))
        gen['avg_entropy'] = avg_entropys
        gen['avg_entropy_of_first_answer'] = avg_entropys[0]

        scores_of_biggest_cluster = gen['scores'][ans_idx]
        scores_of_first_answer = gen['scores'][0]
        gen['std_of_biggest_cluster'] = np.std(scores_of_biggest_cluster)
        gen['std_of_first_answer'] = np.std(scores_of_first_answer)

        gen_answer = gen_answers[ans_idx].replace('\n', '')
        print(gen_answer)
        res_of_biggest_cluster = single_eval(gen_answer)
        res_of_first_answer = single_eval(gen_answers[0].replace('\n', ''))

        print(res_of_biggest_cluster)
        if res_of_biggest_cluster[1] == 2:
            gen['correct_of_biggest_cluster'] = True
        else:
            gen['correct_of_biggest_cluster'] = False
        
        if res_of_first_answer[1] == 2:
            gen['correct_of_first_answer'] = True
        else:
            gen['correct_of_first_answer'] = False
        
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