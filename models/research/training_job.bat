:: From tensorflow/models/research/
SET PIPELINE_CONFIG_PATH=./object_detection/training/models/model/ssd_mobilenet_v2_coco.config
SET MODEL_DIR=./object_detection/training/models/model
SET NUM_TRAIN_STEPS=50000
SET SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python object_detection/model_main.py ^
    --pipeline_config_path=%PIPELINE_CONFIG_PATH% ^
    --model_dir=%MODEL_DIR% ^
    --num_train_steps=%NUM_TRAIN_STEPS% ^
    --sample_1_of_n_eval_examples=%SAMPLE_1_OF_N_EVAL_EXAMPLES% ^
    --alsologtostderr
