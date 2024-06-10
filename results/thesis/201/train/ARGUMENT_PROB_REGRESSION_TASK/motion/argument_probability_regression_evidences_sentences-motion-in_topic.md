# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/evidences_sentences.csv \
    --task ARGUMENT_PROB_REGRESSION_TASK \
    --model_name bert-large-uncased \
    --include_motion True \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 64 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/argument_probability_regression_evidences_sentences-motion-in_topic \
    --run_name argument_probability_regression_evidences_sentences-motion-in_topic \
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
    "test_loss": 0.03374577313661575,
    "test_mean_squared_error": 0.03374577686190605,
    "test_root_mean_squared_error": 0.18370023369789124,
    "test_spearman": 0.8133068494209846,
    "test_pearson": 0.8308568488312505,
    "epochs_trained_total": 5.431007751937985,
    "epochs_trained_best": 2.356852420652333
}
```

