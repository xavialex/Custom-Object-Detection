:: From tensorflow/models/research/
SET INPUT_TYPE=image_tensor
SET PIPELINE_CONFIG_PATH=./object_detection/training/models/model/ssd_mobilenet_v2_coco.config
SET TRAINED_CKPT_PREFIX=./object_detection/training/models/model/best_models/model.ckpt-10228
SET EXPORT_DIR=./object_detection/training/models/model/model_10-07/
python object_detection/export_inference_graph.py ^
    --input_type=%INPUT_TYPE% ^
    --pipeline_config_path=%PIPELINE_CONFIG_PATH% ^
    --trained_checkpoint_prefix=%TRAINED_CKPT_PREFIX% ^
    --output_directory=%EXPORT_DIR%