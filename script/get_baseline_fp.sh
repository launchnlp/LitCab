set -e

model_names=("decapoda-research/llama-7b-hf" "decapoda-research/llama-13b-hf" "meta-llama/Llama-2-13b-hf" "lmsys/vicuna-13b-v1.3" "EleutherAI/gpt-j-6b")

for model_name in "${model_names[@]}"; do
    model_name_postfix=$(echo "$model_name" | cut -d'/' -f2)
    echo "model_name_postfix: ${model_name_postfix}"
    cd ../long_form_generation

    # 1. gen bios
    echo "gen_bio"
    echo "../factuality_prompt/test.${model_name_postfix}.csv"
    if [ ! -f "../factuality_prompt/test.${model_name_postfix}.csv" ]; then
        python prompt_gen.py $model_name
    fi

    # 2. analysis bios
    echo "analysis fp generation"
    if [ ! -f "../factuality_prompt/test.${model_name_postfix}.bio_facts_segs.json" ]; then
        python analysis_fp_generation.py ${model_name}
    fi

    # 3. get correctness
    echo "get correctness"
    gen_file="../factuality_prompt/test.${model_name_postfix}.bio_facts_segs.json"
    if [ ! -f "../factuality_prompt/test.${model_name_postfix}.bio_facts_segs.correctness.json" ]; then
        python factscorer.py --bio_fact_path ${gen_file}
    fi

    # 4. get segment prob
    echo "get segment prob"
    gen_file="../factuality_prompt/test.${model_name_postfix}.bio_facts_segs.correctness.json"
    if [ ! -f "../factuality_prompt/test.${model_name_postfix}.bio_facts_segs.correctness.reeval.json" ]; then
        python get_segment_prob_for_fp.py ${gen_file} ${model_name}
    fi

    # 5. get metrics
    echo "get segment prob"
    gen_file="../factuality_prompt/test.${model_name_postfix}.bio_facts_segs.correctness.reeval.json"
    python get_ece_for_long_form.py ${gen_file} > ../script/log/long_form_${model_name_postfix}.log
done