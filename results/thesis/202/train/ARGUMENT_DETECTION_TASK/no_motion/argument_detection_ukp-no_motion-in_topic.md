# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining.csv \
    --task ARGUMENT_DETECTION_TASK \
    --model_name bert-large-uncased \
    --include_motion False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 64 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/argument_detection_ukp-no_motion-in_topic \
    --run_name argument_detection_ukp-no_motion-in_topic \
    --load_best_model_at_end 1 \
    --save_total_limit 1 \
    --mlflow_env .mlflow \
    --early_stopping_patience 20 \
    --max_len_percentile 99 \
    --seed 202 \
    --result_prefix 202
```

# Results
```json
{
    "test_loss": 0.44235119223594666,
    "test_f1_micro": 0.7938931297709924,
    "test_f1_macro": 0.7929434671821998,
    "test_f1_weighted": 0.7932042126906833,
    "test_precision_micro": 0.7938931297709924,
    "test_precision_macro": 0.7968602012812195,
    "test_precision_weighted": 0.7963880542234781,
    "test_recall_micro": 0.7938931297709924,
    "test_recall_macro": 0.7929079898344589,
    "test_recall_weighted": 0.7938931297709924,
    "test_f1_true": 0.8069660861594867,
    "test_f1_false": 0.7789208482049129,
    "test_precision_true": 0.7714686295127936,
    "test_precision_false": 0.8222517730496454,
    "test_recall_true": 0.8458877786318216,
    "test_recall_false": 0.7399282010370961,
    "epochs_trained_total": 4.041811846689895,
    "epochs_trained_best": 2.0209059233449476
}
```

