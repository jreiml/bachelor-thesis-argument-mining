# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/evidences_sentences_cross_topic.csv \
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
    --output_dir models/argument_probability_regression_evidences_sentences-motion-cross_topic \
    --run_name argument_probability_regression_evidences_sentences-motion-cross_topic \
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
    "test_loss": 0.038620781153440475,
    "test_mean_squared_error": 0.038620784878730774,
    "test_root_mean_squared_error": 0.19652171432971954,
    "test_spearman": 0.775901114840808,
    "test_pearson": 0.7888732593136591,
    "epochs_trained_total": 6.804992199687987,
    "epochs_trained_best": 3.711813927102539
}
```

