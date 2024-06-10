# Command
```bash
python3 src/models/train_model.py \
    --input_dataset_csv data/processed/ukp_sentential_argument_mining.csv data/processed/evidences_sentences.csv \
    --input_dataset_task_head 0 1 \
    --task ARGUMENT_DETECTION_ARGUMENT_PROB_REGRESSION_MULTI_TASK \
    --task_head_loss_weights 1 10.5 \
    --model_name bert-large-uncased \
    --include_motion False \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --logging_epoch_step 0.1 \
    --learning_rate 1e-05 \
    --num_train_epochs 10 \
    --output_dir models/multitask-no_motion-detection_evidence_in_topic \
    --run_name multitask-no_motion-detection_evidence_in_topic \
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
    "test_ARGUMENT_DETECTION_TASK_loss": 0.44539764523506165,
    "test_ARGUMENT_PROB_REGRESSION_TASK_loss": 0.432199627161026,
    "test_loss": 0.8775972723960876,
    "test_ARGUMENT_DETECTION_TASK-f1_micro": 0.7942845958113135,
    "test_ARGUMENT_DETECTION_TASK-f1_macro": 0.7939496025337454,
    "test_ARGUMENT_DETECTION_TASK-f1_weighted": 0.7936487590563179,
    "test_ARGUMENT_DETECTION_TASK-precision_micro": 0.7942845958113134,
    "test_ARGUMENT_DETECTION_TASK-precision_macro": 0.8005034827095032,
    "test_ARGUMENT_DETECTION_TASK-precision_weighted": 0.8024306084652237,
    "test_ARGUMENT_DETECTION_TASK-recall_micro": 0.7942845958113134,
    "test_ARGUMENT_DETECTION_TASK-recall_macro": 0.7967894006273522,
    "test_ARGUMENT_DETECTION_TASK-recall_weighted": 0.7942845958113134,
    "test_ARGUMENT_DETECTION_TASK-f1_true": 0.8022577610536218,
    "test_ARGUMENT_DETECTION_TASK-f1_false": 0.785641444013869,
    "test_ARGUMENT_DETECTION_TASK-precision_true": 0.7472835611636873,
    "test_ARGUMENT_DETECTION_TASK-precision_false": 0.8537234042553191,
    "test_ARGUMENT_DETECTION_TASK-recall_true": 0.8659626320064988,
    "test_ARGUMENT_DETECTION_TASK-recall_false": 0.7276161692482055,
    "test_ARGUMENT_PROB_REGRESSION_TASK-mean_squared_error": 0.04121064767241478,
    "test_ARGUMENT_PROB_REGRESSION_TASK-root_mean_squared_error": 0.20300406217575073,
    "test_ARGUMENT_PROB_REGRESSION_TASK-spearman": 0.7636507811303995,
    "test_ARGUMENT_PROB_REGRESSION_TASK-pearson": 0.7904678072926742,
    "epochs_trained_total": 5.408531583264971,
    "epochs_trained_best": 2.4037918147844315
}
```

