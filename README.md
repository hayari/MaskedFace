# MaskedFace

Detection of masked/unmasked faces using CNN

realized as a mini project for the  **IA 02: Neural Network and Deep Learning  track** at **GoMyCode** https://gomycode.tn/



### Datasets used

I used RFMD and LFW datasets for training my CNN model

download instructions are located in https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset

 

The model did **99,4%** accuracy on test dataset 

The trained model is stored in the '**my_model_128.h5**' file 

### Requirements

- Tensorflow (tested using version 2.3.0)

  ```bash
  $ pip install tensorflow
  ```

- MTCNN (multitask cascaded convolutional networks)

  ```bash
  $ pip install mtcnn
  ```

- opencv2

### How it work

MTCNN is used to detect faces in the images which are sended to my model in order to classify them masked or unmasked



### Usage

to use streaming from the webcam

```bash
$ python streamv2.py
```

