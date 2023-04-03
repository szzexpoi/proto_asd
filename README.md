# Deep Learning to Interpret Autism Spectrum Disorder Behind the Camera

This repository implements an interpretable and generalizable framework for attention-based autism screening. It centers around a prototypical inference process, which is applicable to a variety of model architecture, including Convolutional Neural Network, Recurrent Neural Network, and Visual Transformer.

### Requirements
1. Requirements for Pytorch. We use Pytorch 1.9.0 in our experiments.
2. Requirements for Tensorflow (for monitoring training process only).
3. Python 3.6+
4. Jupyter Notebook
5. You may need to install the OpenCV package (CV2) for Python.

### Experiment
To run the experiment for the leave-one-subject-out evaluation:

```
python main.py --mode train --img_dir IMG_DIR --checkpoint_path CKPT_DIR --lr 1e-4 --alpha 0.01 --beta 0.01 --margin 0.8 --n_proto 10 --model_type MODEL
```

where **IMG_DIR** is the directory storing the image data organized by class labels, **$CKPT_DIR** is the directory for storing checkpoint, **MODEL** specifies the backbone architecture (cnn, rnn, or transformer).

### Analysis
To interpret the rationales behind the model's prediction, here we use our framework with CNN as an example, and provide code for measuring the importance of different prototypes. The results can be obtained with two steps:

1. Computing the gradient-based importance for each sample with Grad-CAM:
```
python main.py --mode analysis_single --img_dir IMG_DIR --weights CKPT_DIR --n_proto 10 --model_type cnn --save_dir SAVE_DIR
```
where **SAVE_DIR** is the directory storing intermediate results.

2. Computing the overall importance for different prototypes. This can be done by following our Jupyter Notebook.

### Reference
If you use our code or data, please cite our paper:
```
TBD
```
