# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv data/processed/ukp_sentential_argument_mining.csv \
    --input_dataset_task_head 0 1 \
    --task STRENGTH_REGRESSION_ARGUMENT_DETECTION_MULTI_TASK \
    --task_head_loss_weights 14.8 1 \
    --model_name bert-large-uncased \
    --include_motion True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-motion-strength_detection_wa_in_topic \
    --run_name multitask-motion-strength_detection_wa_in_topic \
    --load_best_model_at_end 1 \
    --save_total_limit 1 \
    --mlflow_env .mlflow \
    --early_stopping_patience 30 \
    --max_len_percentile 99 \
    --seed 200 \
    --result_prefix 200
```

# Results
```json
{
    "test_STRENGTH_REGRESSION_TASK_loss": 0.37575480341911316,
    "test_ARGUMENT_DETECTION_TASK_loss": 0.4405956566333771,
    "test_loss": 0.8163504600524902,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.025388840585947037,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.15933875739574432,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.5048162761745814,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5532767148920541,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.8028968486983755,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.802896365413948,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.8029333970831987,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.8028968486983754,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.8141973081713165,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8258006275296296,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.8028968486983754,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.8144337550889929,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.8028968486983754,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8025877278964908,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.8032050029314052,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.7174903610234841,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.910904255319149,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.9105871886120996,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.7182803215658861,
    "epochs_trained_total": 5.781979082864039,
    "epochs_trained_best": 2.7388321971461234
}
```

