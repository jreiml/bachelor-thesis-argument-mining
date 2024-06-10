# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv data/processed/ukp_sentential_argument_mining_cross_topic.csv data/processed/evidences_sentences_cross_topic.csv \
    --input_dataset_task_head 0 1 2 \
    --task ALL_MULTI_TASK \
    --task_head_loss_weights 18.2 1 7.6 \
    --model_name bert-large-uncased \
    --include_motion False \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-no_motion-strength_detection_evidence_wa_cross_topic \
    --run_name multitask-no_motion-strength_detection_evidence_wa_cross_topic \
    --load_best_model_at_end 1 \
    --save_total_limit 1 \
    --mlflow_env .mlflow \
    --early_stopping_patience 30 \
    --max_len_percentile 99 \
    --seed 202 \
    --result_prefix 202
```

# Results
```json
{
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.4123542606830597,
    "test_STRENGTH_REGRESSION_TASK_loss": 0.545863151550293,
    "test_ARGUMENT_DETECTION_TASK_loss": 0.4703531563282013,
    "test_loss": 1.428570568561554,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.02999248169362545,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.1731833815574646,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.4611616667867516,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5133279370091404,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.774858528698464,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.7662791723202171,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.781012497408599,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.774858528698464,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.766730572597137,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8161048730932579,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.774858528698464,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.7982334447278163,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.774858528698464,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8110583446404342,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.7215,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.9167944785276073,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.6166666666666667,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.7271897810218978,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.869277108433735,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.05416124686598778,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.23272569477558136,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.6932750986895382,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.69069808314694,
    "epochs_trained_total": 4.13550135501355,
    "epochs_trained_best": 1.109524753784123
}
```

