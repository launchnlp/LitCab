set -e

datasets=("NQ" "truthfulqa" "triviaqa" "wikiqa" "sciq")
# model_name="meta-llama/Llama-2-7b-hf"
model_name=$1
num_samples=10
model_name_postfix=$(echo "$model_name" | cut -d'/' -f2)
for dataset in "${datasets[@]}"; do
    echo "dataset: $dataset"
    
    cd ../src

    # 1. collect logits
    echo "collecting logits"
    # if [[ "$dataset" != "triviaqa" ]]; then
    python collect_logits_for_training_additional_layer.py "${model_name}" "${dataset}"
    # fi

    rm -rf ../${dataset}/calibrator
    if [ ! -d "../${dataset}/calibrator" ]; then
        mkdir "../${dataset}/calibrator"
    fi
    
    # 1. train model
    echo "training model"
    # python train_additional_layer.py "${dataset}" "${model_name}"
    python train_additional_layer.py "${dataset}" "${model_name}"

    # extract 1000 qas
    gen_file="../${dataset}/test${model_name_postfix}temperature_1topp_1.0_num_demos_15_answers.reeval.semantic_uncertainty.1000.gpt4_correctness.jsonl"
    python extract_1000_qas.py "${gen_file}"

    # 2. reevaluate model generations
    echo "reevaluating model generations"
    model_name_postfix=$(echo "$model_name" | cut -d'/' -f2)
    gen_file="../${dataset}/test${model_name_postfix}temperature_1topp_1.0_num_demos_15_answers_1000.csv"
    echo "$gen_file"
    # wc -l "../$dataset/test.txt"
    python reeval4llamagenerated_answer_with_addition_layer.py "$gen_file" "$model_name" "$dataset"

    # 3. get semantic uncertainty
    echo "getting semantic uncertainty"
    gen_file="../${dataset}/test${model_name_postfix}temperature_1topp_1.0_num_demos_15_answers_1000.reeval_with_addition_layer.jsonl"
    python get_semantic_uncertainty.py "$gen_file"

    # 4. correctness
    echo "getting correctness"
    gen_file="../${dataset}/test${model_name_postfix}temperature_1topp_1.0_num_demos_15_answers_1000.reeval_with_addition_layer.semantic_uncertainty.jsonl"
    gen_file_with_correctness="../${dataset}/test${model_name_postfix}temperature_1topp_1.0_num_demos_15_answers.reeval.semantic_uncertainty.1000.gpt4_correctness.jsonl"
    python copy_correctness.py "$gen_file" "$gen_file_with_correctness"

    # 5. get metrics
    cd ../script
    gpt_depolyment=gpt4
    gen_file="../${dataset}/test${model_name_postfix}temperature_1topp_1.0_num_demos_15_answers_1000.reeval_with_addition_layer.semantic_uncertainty.gpt4_correctness.jsonl"
    bash get_metrics.sh "$gen_file" "score_of_first_answer" > log/${dataset}_${model_name_postfix}_${gpt_depolyment}_correctness_additional_layer.log
    # exit
done