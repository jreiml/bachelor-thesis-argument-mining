# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_cross_topic_wa.csv data/processed/evidences_sentences_cross_topic.csv \
    --input_dataset_task_head 0 1 \
    --task STRENGTH_REGRESSION_ARGUMENT_PROB_REGRESSION_MULTI_TASK \
    --task_head_loss_weights 1.8 1 \
    --model_name bert-large-uncased \
    --include_motion True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-motion-strength_evidence_wa_cross_topic \
    --run_name multitask-motion-strength_evidence_wa_cross_topic \
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
    "test_STRENGTH_REGRESSION_TASK_loss": 0.05182301253080368,
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.037634722888469696,
    "test_loss": 0.08945773541927338,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.02875383198261261,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.16956955194473267,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.46716909531877854,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5240702564803009,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.0376347191631794,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.19399669766426086,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.786487008431826,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.7940487280173886,
    "epochs_trained_total": 6.1187355435620665,
    "epochs_trained_best": 3.1095213418102308
}
```

