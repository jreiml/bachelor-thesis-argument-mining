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
    --seed 200 \
    --result_prefix 200
```

# Results
```json
{
    "test_loss": 0.4338406026363373,
    "test_f1_micro": 0.8038755137991779,
    "test_f1_macro": 0.8033857083205438,
    "test_f1_weighted": 0.8032185982160687,
    "test_precision_micro": 0.8038755137991779,
    "test_precision_macro": 0.8090909345146207,
    "test_precision_weighted": 0.8098509706992832,
    "test_recall_micro": 0.8038755137991779,
    "test_recall_macro": 0.8049588737959446,
    "test_recall_weighted": 0.8038755137991779,
    "test_f1_true": 0.8131991051454138,
    "test_f1_false": 0.7935723114956738,
    "test_precision_true": 0.7644584647739222,
    "test_precision_false": 0.8537234042553191,
    "test_recall_true": 0.8685782556750299,
    "test_recall_false": 0.7413394919168591,
    "epochs_trained_total": 4.041811846689895,
    "epochs_trained_best": 2.0209059233449476
}
```

