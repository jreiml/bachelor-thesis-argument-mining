# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining.csv \
    --task ARGUMENT_DETECTION_TASK \
    --model_name bert-large-uncased \
    --include_motion True \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 64 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/argument_detection_ukp-motion-in_topic \
    --run_name argument_detection_ukp-motion-in_topic \
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
    "test_loss": 0.4478347599506378,
    "test_f1_micro": 0.7933059307105107,
    "test_f1_macro": 0.7932431873123129,
    "test_f1_weighted": 0.7937895483640368,
    "test_precision_micro": 0.7933059307105108,
    "test_precision_macro": 0.8063518897614307,
    "test_precision_weighted": 0.8232875987471306,
    "test_recall_micro": 0.7933059307105108,
    "test_recall_macro": 0.8092857073392777,
    "test_recall_weighted": 0.7933059307105108,
    "test_f1_true": 0.7896414342629482,
    "test_f1_false": 0.7968449403616775,
    "test_precision_true": 0.6947073256221521,
    "test_precision_false": 0.9179964539007093,
    "test_recall_true": 0.9146285186894324,
    "test_recall_false": 0.7039428959891231,
    "epochs_trained_total": 4.041811846689895,
    "epochs_trained_best": 2.0209059233449476
}
```

