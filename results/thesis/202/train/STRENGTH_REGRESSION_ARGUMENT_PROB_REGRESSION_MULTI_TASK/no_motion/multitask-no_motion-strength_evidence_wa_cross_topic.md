# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv data/processed/evidences_sentences_cross_topic.csv \
    --input_dataset_task_head 0 1 \
    --task STRENGTH_REGRESSION_ARGUMENT_PROB_REGRESSION_MULTI_TASK \
    --task_head_loss_weights 2.4 1 \
    --model_name bert-large-uncased \
    --include_motion False \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-no_motion-strength_evidence_wa_cross_topic \
    --run_name multitask-no_motion-strength_evidence_wa_cross_topic \
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
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.050467297434806824,
    "test_STRENGTH_REGRESSION_TASK_loss": 0.06907471269369125,
    "test_loss": 0.11954201012849808,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.028841601684689522,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.1698281466960907,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.4737796067736991,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5236827090898625,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.050467297434806824,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.22464928030967712,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.7064935129946242,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.7081161018873483,
    "epochs_trained_total": 4.914417887432537,
    "epochs_trained_best": 1.905590609412616
}
```

