# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/arg_quality_rank_30k_wa.csv data/processed/evidences_sentences.csv \
    --input_dataset_task_head 0 1 \
    --task STRENGTH_REGRESSION_ARGUMENT_PROB_REGRESSION_MULTI_TASK \
    --task_head_loss_weights 1.6 1 \
    --model_name bert-large-uncased \
    --include_motion True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-motion-strength_evidence_wa_in_topic \
    --run_name multitask-motion-strength_evidence_wa_in_topic \
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
    "test_STRENGTH_REGRESSION_TASK_loss": 0.04112638905644417,
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.031925179064273834,
    "test_loss": 0.073051568120718,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.025677192956209183,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.1602410525083542,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.5110533160656546,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5610223599756883,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.031925179064273834,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.17867618799209595,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.8257618447941133,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.8466746617303269,
    "epochs_trained_total": 7.835616438356165,
    "epochs_trained_best": 4.821917808219178
}
```

