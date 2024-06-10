# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining_cross_topic.csv \
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
    --output_dir models/argument_detection_ukp-motion-cross_topic \
    --run_name argument_detection_ukp-motion-cross_topic \
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
    "test_loss": 0.6222579479217529,
    "test_f1_micro": 0.7021018593371059,
    "test_f1_macro": 0.6663349544411131,
    "test_f1_weighted": 0.731951770089811,
    "test_precision_micro": 0.7021018593371059,
    "test_precision_macro": 0.6863821378008494,
    "test_precision_weighted": 0.8607068109566485,
    "test_recall_micro": 0.7021018593371059,
    "test_recall_macro": 0.7907204637467795,
    "test_recall_weighted": 0.7021018593371059,
    "test_f1_true": 0.7755785627283801,
    "test_f1_false": 0.5570913461538461,
    "test_precision_true": 0.9766104294478528,
    "test_precision_false": 0.39615384615384613,
    "test_recall_true": 0.6431818181818182,
    "test_recall_false": 0.9382591093117408,
    "epochs_trained_total": 3.27007299270073,
    "epochs_trained_best": 1.2262773722627738
}
```

