
MODEL_GEN_FILE=$1
if echo "$MODEL_GEN_FILE" | grep -q "investigate_noise"; then
    FIRST_FACTORS=("score_of_first_answer" "avg_entropy_of_first_answer" "std_of_first_answer" "max_score_of_first_answer" "min_score_of_first_answer" "D_var" "semantic_entropy" "num_clusters" "proportion_of_biggest_clusters" "mean_abs_diffs_of_first_answer" "std_of_mean_abs_diffs_of_first_answer"  "lex_sim" "lex_sim&score_of_first_answer")
else
    FIRST_FACTORS=("score_of_first_answer" "avg_entropy_of_first_answer" "std_of_first_answer" "std_normalized_scores_of_each_ensemble" "D_var" "semantic_entropy" "num_clusters" "proportion_of_biggest_clusters" "lex_sim" "lex_sim&score_of_first_answer")
fi
if echo "$MODEL_GEN_FILE" | grep -q "QG"; then
    FIRST_FACTORS=("score_of_first_answer" "avg_entropy_of_first_answer" "std_of_first_answer" "D_var" "semantic_entropy" "num_clusters" "proportion_of_biggest_clusters" "lex_sim" "lex_sim&score_of_first_answer" "QG_score_of_first_answer" "avg_QG_entropy_of_first_answer" "std_of_QG_scores_of_first_answer" "D_var_QG" "avg_QG_score" "score_of_first_answer&QG_score_of_first_answer")
    # FIRST_FACTORS=("score_of_first_answer" "avg_entropy_of_first_answer" "std_of_first_answer" "QG_score_of_first_answer" "avg_QG_entropy_of_first_answer" "std_of_QG_scores_of_first_answer" "D_var_QG" "avg_QG_score" "score_of_first_answer&QG_score_of_first_answer")
fi

# FIRST_FACTORS=("score_of_first_answer")
# FIRST_FACTORS=("calibrator_score")
# FIRST_FACTORS=("verbalization_prob")
FIRST_FACTORS=($2)

if echo "$MODEL_GEN_FILE" | grep -q "factualityprompt"; then
    for first_factor in "${FIRST_FACTORS[@]}"; do
        echo "first factor: $first_factor"
        echo "AUROC:"
        if [ "$first_factor" = "semantic_entropy" ] || [ "$first_factor" = "num_clusters" ] || [ "$first_factor" = "proportion_of_biggest_clusters" ] || [ "$first_factor" = "lex_sim" ];then
            second_factor="correct_of_biggest_cluster"
        else
            second_factor="correct_of_first_answer"
        fi
        python ../evaluate/get_auroc.py ${MODEL_GEN_FILE} ${first_factor} ${second_factor}

        echo "ECE:"
        if [ "$first_factor" = "semantic_entropy" ] || [ "$first_factor" = "num_clusters" ] || [ "$first_factor" = "proportion_of_biggest_clusters" ] || [ "$first_factor" = "lex_sim" ];then
            second_factor="correct_of_biggest_cluster"
        else
            second_factor="correct_of_first_answer"
        fi
        python ../evaluate/get_ECE.py ${MODEL_GEN_FILE} ${first_factor} ${second_factor}

        echo "brier:"
        python ../evaluate/get_brier_score.py ${MODEL_GEN_FILE} ${first_factor} ${second_factor}
    done
else
    for first_factor in "${FIRST_FACTORS[@]}"; do
        echo "first factor: $first_factor"

        # echo "correlation:"
        # if [ "$first_factor" = "semantic_entropy" ] || [ "$first_factor" = "num_clusters" ] || [ "$first_factor" = "proportion_of_biggest_clusters" ] || [ "$first_factor" = "lex_sim" ];then
        #     # second_factor="rougeL_of_biggest_cluster"
        #     second_factor="rougeL_first_answer"
        # else
        #     second_factor="rougeL_first_answer"
        # fi
        # python ../evaluate/calculate_correlation.py ${MODEL_GEN_FILE} ${first_factor} ${second_factor}

        echo "AUROC:"
        if [ "$first_factor" = "semantic_entropy" ] || [ "$first_factor" = "num_clusters" ] || [ "$first_factor" = "proportion_of_biggest_clusters" ] || [ "$first_factor" = "lex_sim" ];then
            # second_factor="rouge_correct_of_biggest_cluster"
            second_factor="rouge_correct_first_answer"
        else
            second_factor="rouge_correct_first_answer"
        fi
        python ../evaluate/get_auroc.py ${MODEL_GEN_FILE} ${first_factor} ${second_factor}

        echo "ECE:"
        if [ "$first_factor" = "semantic_entropy" ] || [ "$first_factor" = "num_clusters" ] || [ "$first_factor" = "proportion_of_biggest_clusters" ] || [ "$first_factor" = "lex_sim" ];then
            # second_factor="rouge_correct_of_biggest_cluster"
            second_factor="rouge_correct_first_answer"
        else
            second_factor="rouge_correct_first_answer"
        fi
        python ../evaluate/get_ECE.py ${MODEL_GEN_FILE} ${first_factor} ${second_factor}

        echo "brier:"
        python ../evaluate/get_brier_score.py ${MODEL_GEN_FILE} ${first_factor} ${second_factor}

        echo "acc@q and cov@p:"
        for p in 0.4 0.5 0.6 0.7 0.8 0.9; do
        echo "q: 0.5" "p: $p"
            python ../evaluate/get_acc_q.py ${MODEL_GEN_FILE} ${first_factor} ${second_factor} 0.5 ${p}
        done
    done
fi
