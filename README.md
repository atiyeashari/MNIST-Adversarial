# MNIST-Adversarial

In this project, first, I built a multi-layer convolutional neural network using [Deep MNIST for Experts](https://www.tensorflow.org/get_started/mnist/pros#deep-mnist-for-experts) with 99.15% accuracy. Then using [this code](https://github.com/andrwc/Adversarial-MNIST) based on [Breaking Linear Classifiers on ImageNet](http://karpathy.github.io/2015/03/30/breaking-convnets/) and [Goodfellow et al](https://arxiv.org/abs/1412.6572), I generated adversarial images. Specifically, 10 images of digit ‘2’ which are correctly classified as ‘2’ by the trained model and modified so the network incorrectly classifies them as 6.

### Prerequisites

What things you need to have installed.

```
tensorflow
numpy
matplotlib
pandas
```

## Running the tests
First, run the 
