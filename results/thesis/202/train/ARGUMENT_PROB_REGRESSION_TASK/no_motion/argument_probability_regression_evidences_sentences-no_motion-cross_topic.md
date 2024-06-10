# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/evidences_sentences_cross_topic.csv \
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
    --output_dir models/argument_probability_regression_evidences_sentences-no_motion-cross_topic \
    --run_name argument_probability_regression_evidences_sentences-no_motion-cross_topic \
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
    "test_loss": 0.05151403695344925,
    "test_mean_squared_error": 0.05151403322815895,
    "test_root_mean_squared_error": 0.22696703672409058,
    "test_spearman": 0.7037871152036735,
    "test_pearson": 0.7032062739620617,
    "epochs_trained_total": 6.084243369734789,
    "epochs_trained_best": 2.9905603003781165
}
```

