# Custom Object Detection

This project aims to easily use the TF Object Detection library to quicly use pretrained models for Transfer Learning purposes. A new dataset must be created using a labeling tool. After that, this dataset should be used to retrain the chosen pretrained model on it. Finally, with presumably less training time, a new model would converge to be used to predict new scenarios involving the new classes of interest.

## Dependencies

## Use

This section enumerates the steps to train a model. The next directory structure (in *research -> object_detection -> training*) is recommended:

```
+data
  -label_map file
  -train TFRecord file
  -eval TFRecord file
+models
  + model
    -pipeline config file
    +train
    +eval
```

1. Generate a dataset containig instances of the new classes to be detected by the model
2. Tag the images using any labeling tool available ([Voot](https://github.com/microsoft/VoTT) allows to export directly in TFRecord format) 
3. Create a new *label_map.pbtxt* file that'll look like this:
```
item {
  id: 1
  name: 'my_first_class'
}
item {
  id: 2
  name: 'my_second_class'
}
.
.
.
```
4. Create a [pipeline config file](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md). Change all parameters regarding the location of training and validation sets, pretrained models checkpoints (if any), etc.  
>> In order to apply Transfer Learning and accelerate learning, a pretrained model from [TF Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) will be needed.  
5. From the *research* directory, launch *training_job.bat* to start the training. Modify the paths to correctly point to the configuration file and to the directory (*model*) where to save the checkpoints achieved during training. 
6. To visualize the evolution of the model perfomrance, use the TensorBoard utility as following:
```
tensorboard --logdir=path_to_training_and_eval_logs
```

Once the model achives a desired checkpoint, it should be exported into a TensorFlow model. Make use of the *export_model.bat* script in the reseach directory, configuring again the paths for the configuration file, the checkpoint of interest and the location where the model will be exported). Then, the model is ready to be used for inference. Refer to [this project](https://github.com/xavialex/object-detection-inference) to see how to build an application that takes profit of this model.
