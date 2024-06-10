# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv \
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
    --output_dir models/argument_detection_ukp-no_motion-cross_topic \
    --run_name argument_detection_ukp-no_motion-cross_topic \
    --load_best_model_at_end 1 \
    --save_total_limit 1 \
    --mlflow_env .mlflow \
    --early_stopping_patience 20 \
    --max_len_percentile 99 \
    --seed 201 \
    --result_prefix 201
```

# Results
```json
{
    "test_loss": 0.7781496047973633,
    "test_f1_micro": 0.6459175424413904,
    "test_f1_macro": 0.589666806775194,
    "test_f1_weighted": 0.6939394586979033,
    "test_precision_micro": 0.6459175424413904,
    "test_precision_macro": 0.6277050233338577,
    "test_precision_weighted": 0.8584875415770715,
    "test_recall_micro": 0.6459175424413904,
    "test_recall_macro": 0.7407273329313736,
    "test_recall_weighted": 0.6459175424413904,
    "test_f1_true": 0.7415929203539824,
    "test_f1_false": 0.4377406931964057,
    "test_precision_true": 0.963957055214724,
    "test_precision_false": 0.2914529914529915,
    "test_recall_true": 0.6025886864813039,
    "test_recall_false": 0.8788659793814433,
    "epochs_trained_total": 3.883211678832117,
    "epochs_trained_best": 1.8394160583941606
}
```

