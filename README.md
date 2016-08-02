## EasyCNN
Easy convolution network.

## Features
* Basic layer : data layer,convolution layer,pooling layer,full connect layer,softmax layer.
* Loss function : Cross Entropy,MSE.
* Optimize method : SGD

## Demo
* mnist demo :  ![examples/mnist_train_test.cpp](./examples/mnist_train_test.cpp "mnist_train_test.cpp")

## TODO
* fix train error when batch > 1 issue.
* add load & save model function.
* add more layer,such as batch normalization layer,dropout layer,etc.
* port to other platforms,such as linux,mac,android,iOS,etc.
* optimize network train/test speed.
