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
    --seed 201 \
    --result_prefix 201
```

# Results
```json
{
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.033689435571432114,
    "test_STRENGTH_REGRESSION_TASK_loss": 0.040563855320215225,
    "test_loss": 0.07425329089164734,
    "test_STRENGTH_REGRESSION_TASK-mean_squared_error": 0.025318585336208344,
    "test_STRENGTH_REGRESSION_TASK-root_mean_squared_error": 0.1591181457042694,
    "test_STRENGTH_REGRESSION_TASK-spearman": 0.507159860520203,
    "test_STRENGTH_REGRESSION_TASK-pearson": 0.552307441417677,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.033689435571432114,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.18354682624340057,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.8155029858332022,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.8352348272714485,
    "epochs_trained_total": 5.223744292237443,
    "epochs_trained_best": 2.2100456621004567
}
```

