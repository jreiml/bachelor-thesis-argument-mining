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
    --seed 200 \
    --result_prefix 200
```

# Results
```json
{
    "test_STRENGTH_REGRESSION_TASK_loss": 0.051288802176713943,
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.039947595447301865,
    "test_loss": 0.09123639762401581,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.028493782505393028,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.16880100965499878,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.4719639273487973,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5321512947753734,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.039979465305805206,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.19994865357875824,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.7756073849981402,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.7831172371450648,
    "epochs_trained_total": 6.519660755589823,
    "epochs_trained_best": 3.5105865607022126
}
```

