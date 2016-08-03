## EasyCNN
Easy convolution network.

## license
This project is released under the [WTFPL LICENSE](http://www.wtfpl.net/ "WTFPL LICENSE").

## Features
* All in one : without any dependency,pure c++ implemented.
* Basic layer : data layer,convolution layer,pooling layer,full connect layer,softmax layer.
* Loss function : Cross Entropy,MSE.
* Optimize method : SGD.

## Demo
* mnist demo :  [examples/mnist_train_test.cpp](./examples/mnist_train_test.cpp "mnist_train_test.cpp")

## TODO
* fix train error when batch > 1 issue.
* add load & save model function.(done)
* add more layer,such as batch normalization layer,dropout layer,etc.
* port to other platforms,such as linux,mac,android,iOS,etc.
* optimize network train/test speed.
* add more optimize method.
* add unit test
* add license.(done)
