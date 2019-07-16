# Custom Object Detection

This project aims to easily use the TF Object Detection library to quicly use pretrained models for Transfer Learning purposes. A new dataset must be created using a labeling tool. After that, this dataset should be used to retrain the chosen pretrained model on it. Finally, with presumably less training time, a new model would converge to be used to predict new scenarios involving the new classes of interest.

## Dependencies

## Use

1. Generate a dataset containig instances of the new classes to be detected by the model
2. Tag the images using any labeling tool available ([Voot](https://github.com/microsoft/VoTT) allows to export directly in TFRecord format) 
3. 