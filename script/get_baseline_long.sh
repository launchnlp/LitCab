set -e

model_names=("decapoda-research/llama-7b-hf" "decapoda-research/llama-13b-hf" "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf")


for model_name in "${model_names[@]}"; do
    model_name_postfix=$(echo "$model_name" | cut -d'/' -f2)
    cd ../long_form_generation

    # 1. gen bios
    echo "gen_bio"
    python gen_bio.py $model_name

    # 2. analysis bios
    echo "analysis bios"
    if [ ! -f "../name_bio/prompt_entities.${model_name_postfix}.bio_facts_segs.json" ]; then
        python analysis_bio.py ${model_name}
    fi

    # 3. get correctness
    echo "get correctness"
    gen_file="../name_bio/prompt_entities.${model_name_postfix}.bio_facts_segs.json"
    if [ ! -f "../name_bio/prompt_entities.${model_name_postfix}.bio_facts_segs.correctness.json" ]; then
        python factscorer.py --bio_fact_path ${gen_file}
    fi

    # 4. get segment prob
    echo "get segment prob"
    gen_file="../name_bio/prompt_entities.${model_name_postfix}.bio_facts_segs.correctness.json"
    python get_segment_prob.py ${gen_file} ${model_name}

    # 5. get metrics
    echo "get segment prob"
    gen_file="../name_bio/prompt_entities.${model_name_postfix}.bio_facts_segs.correctness.reeval.json"
    # change to get_acc_q_for_long_form.py or get_brier_score_for_long_form.py for other metrics
    python get_ece_for_long_form.py ${gen_file} > ../script/log/long_form_${model_name_postfix}.log
done