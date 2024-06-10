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
    --seed 201 \
    --result_prefix 201
```

# Results
```json
{
    "test_loss": 0.4327073097229004,
    "test_f1_micro": 0.8036797807790175,
    "test_f1_macro": 0.8032810957173334,
    "test_f1_weighted": 0.8030436180936347,
    "test_precision_micro": 0.8036797807790175,
    "test_precision_macro": 0.8094722054425726,
    "test_precision_weighted": 0.8108014553234889,
    "test_recall_micro": 0.8036797807790175,
    "test_recall_macro": 0.805466157565861,
    "test_recall_weighted": 0.8036797807790175,
    "test_f1_true": 0.8121371043266529,
    "test_f1_false": 0.794425087108014,
    "test_precision_true": 0.7599018576936558,
    "test_precision_false": 0.8590425531914894,
    "test_recall_true": 0.8720836685438456,
    "test_recall_false": 0.7388486465878765,
    "epochs_trained_total": 3.8397212543554007,
    "epochs_trained_best": 1.818815331010453
}
```

