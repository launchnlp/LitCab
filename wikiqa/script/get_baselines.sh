set -e

# datasets=("NQ" "sciq" "truthfulqa" "triviaqa" "wikiqa")
# datasets=("triviaqa" "truthfulqa" "wikiqa")
datasets=("NQ" "sciq")
# datasets=("truthfulqa")
# model_name="meta-llama/Llama-2-7b-hf"
model_name=$1
num_samples=1
model_name_postfix=$(echo "$model_name" | cut -d'/' -f2)
for dataset in "${datasets[@]}"; do
    echo "dataset: $dataset"
    model_name_postfix=$(echo "$model_name" | cut -d'/' -f2)
    gen_file="../../ICLR_2024/${dataset}/test${model_name_postfix}temperature_1topp_1.0_num_demos_15_answers.${num_samples}.csv"
    
    cd ../src
    # 1. gen model answers
    echo "generating model answers"
    # if gen_file exists, skip
    if [ -f "$gen_file" ]; then
        echo "$gen_file exists, skipping"
    else
        python gen_model_answers.py $model_name $dataset $num_samples
    fi

    # 2. reevaluate model generations
    echo "reevaluating model generations"
    tmp_gen_file="../../ICLR_2024/${dataset}/test${model_name_postfix}temperature_1topp_1.0_num_demos_15_answers.${num_samples}.reeval.jsonl"
    
    # echo "$gen_file"
    # wc -l "../$dataset/test.txt"
    if [ -f "$tmp_gen_file" ]; then
        echo "$tmp_gen_file exists, skipping"
    else
        python reeval4llamagenerated_answer.py "$gen_file" "$model_name" "../$dataset/train.txt"
    fi

    gen_file="../../ICLR_2024/${dataset}/test${model_name_postfix}temperature_1topp_1.0_num_demos_15_answers.${num_samples}.reeval.jsonl"
    # 3. get semantic uncertainty
    echo "getting semantic uncertainty"
    
    python get_semantic_uncertainty.py "$gen_file"


    # # 4. rouge correctness
    # # get correctness
    # echo "getting correctness"
    # gen_file="../../ICLR_2024/${dataset}/test${model_name_postfix}temperature_1topp_1.0_num_demos_15_answers.reeval.semantic_uncertainty.jsonl"
    # python get_rouge_correctness.py "$gen_file" "../$dataset/test.txt"

    # # 5. get metrics
    # cd ../script
    # gen_file="../../ICLR_2024/${dataset}/test${model_name_postfix}temperature_1topp_1.0_num_demos_15_answers.reeval.semantic_uncertainty.rouge_correctness.jsonl"
    # bash get_metrics.sh "$gen_file" > log/${dataset}_${model_name_postfix}.log


    # # 4. GPT correctness
    echo "getting GPT correctness"
    gen_file="../../ICLR_2024/${dataset}/test${model_name_postfix}temperature_1topp_1.0_num_demos_15_answers.${num_samples}.reeval.semantic_uncertainty.jsonl"
    new_gen_file=$(echo "$gen_file" | sed 's/semantic_uncertainty/semantic_uncertainty.1000/g')
    shuf -n 1000 "$gen_file" > "${new_gen_file}"
    # if dataset is NQ or triviaqa, use gpt4, else gpt35
    # if [[ "$dataset" == "NQ" || "$dataset" == "triviaqa" ]]; then
    #     gpt_depolyment="gpt4"
    # else
    #     gpt_depolyment="gpt35"
    # fi
    gpt_depolyment="gpt4"
    python get_gpt_correctness.py "${new_gen_file}" "../$dataset/test.txt" "$gpt_depolyment"

    # 5. get metrics
    cd ../script
    gen_file="../../ICLR_2024/${dataset}/test${model_name_postfix}temperature_1topp_1.0_num_demos_15_answers.${num_samples}.reeval.semantic_uncertainty.1000.${gpt_depolyment}_correctness.jsonl"
    bash get_metrics.sh "$gen_file" "score_of_first_answer" > log/${dataset}_${model_name_postfix}_${gpt_depolyment}_correctness.log
done