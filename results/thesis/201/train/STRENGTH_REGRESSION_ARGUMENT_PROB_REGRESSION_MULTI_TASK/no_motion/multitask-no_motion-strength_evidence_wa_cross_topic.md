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
    --seed 201 \
    --result_prefix 201
```

# Results
```json
{
    "test_STRENGTH_REGRESSION_TASK_loss": 0.06953764706850052,
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.05293649435043335,
    "test_loss": 0.12247414141893387,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.02897402085363865,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.1702175736427307,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.4742563992539776,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.5262527624176452,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.05284031480550766,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.22987021505832672,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.7037677864991709,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.7006438198491488,
    "epochs_trained_total": 4.51349267540478,
    "epochs_trained_best": 1.5044975584682603
}
```

