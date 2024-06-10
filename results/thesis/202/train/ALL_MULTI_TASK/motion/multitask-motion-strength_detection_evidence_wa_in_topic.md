# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv data/processed/ukp_sentential_argument_mining.csv data/processed/evidences_sentences.csv \
    --input_dataset_task_head 0 1 2 \
    --task ALL_MULTI_TASK \
    --task_head_loss_weights 14.8 1 9.25 \
    --model_name bert-large-uncased \
    --include_motion True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-motion-strength_detection_evidence_wa_in_topic \
    --run_name multitask-motion-strength_detection_evidence_wa_in_topic \
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
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.297335147857666,
    "test_STRENGTH_REGRESSION_TASK_loss": 0.37125951051712036,
    "test_ARGUMENT_DETECTION_TASK_loss": 0.4835740029811859,
    "test_loss": 1.1521686613559723,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.025074439123272896,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.15834911167621613,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.5122610099413565,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5624565886244083,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.8140536308475241,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.8139198267446002,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.8136043615968306,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.814053630847524,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.8221462010873213,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8265245933611144,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.814053630847524,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.8190225756640472,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.814053630847524,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.818909645444148,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.8089300080450522,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.7528916929547844,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.8914007092198581,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.897618052653573,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.7404270986745214,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.032180678099393845,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.17938974499702454,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.8179188788489843,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.8450022733236534,
    "epochs_trained_total": 6.94385593220339,
    "epochs_trained_best": 3.92478813559322
}
```

