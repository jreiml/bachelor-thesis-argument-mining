# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/evidences_sentences.csv \
    --task ARGUMENT_PROB_REGRESSION_TASK \
    --model_name bert-large-uncased \
    --include_motion False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 64 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/argument_probability_regression_evidences_sentences-no_motion-in_topic \
    --run_name argument_probability_regression_evidences_sentences-no_motion-in_topic \
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
    "test_loss": 0.03915039822459221,
    "test_mean_squared_error": 0.03915039822459221,
    "test_root_mean_squared_error": 0.1978645920753479,
    "test_spearman": 0.785324923397957,
    "test_pearson": 0.8107006170914328,
    "epochs_trained_total": 8.607751937984496,
    "epochs_trained_best": 5.533554817275748
}
```

