# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv data/processed/ukp_sentential_argument_mining.csv data/processed/evidences_sentences.csv \
    --input_dataset_task_head 0 1 2 \
    --task ALL_MULTI_TASK \
    --task_head_loss_weights 14.8 1 10.5 \
    --model_name bert-large-uncased \
    --include_motion False \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-no_motion-strength_detection_evidence_wa_in_topic \
    --run_name multitask-no_motion-strength_detection_evidence_wa_in_topic \
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
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.3967791497707367,
    "test_STRENGTH_REGRESSION_TASK_loss": 0.3854958415031433,
    "test_ARGUMENT_DETECTION_TASK_loss": 0.49059221148490906,
    "test_loss": 1.272867202758789,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.02603919804096222,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.16136665642261505,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.49296168351771186,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5397687570835872,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.8044627128596594,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.8042510492486727,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.8039222147101753,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.8044627128596594,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.8117036657941249,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8148693085343183,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.8044627128596594,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.8082519774323801,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.8044627128596594,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8106878908470722,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.7978142076502732,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.7497371188222923,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.8736702127659575,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.8824257425742574,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.7340782122905027,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.03788954019546509,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.19465236365795135,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.7836971480727539,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.8091927778530161,
    "epochs_trained_total": 6.94385593220339,
    "epochs_trained_best": 3.92478813559322
}
```

