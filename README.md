## EasyCNN
Easy convolution network.

## Features
* All in one : without any dependency,pure c++ implemented.
* Basic layer : data layer,convolution layer,pooling layer,full connect layer,softmax layer,activation layers(sigmod,tanh,RELU)
* Loss function : Cross Entropy,MSE.
* Optimize method : SGD.

## Examples
* mnist demo , with ConvNet and MLP net,  [examples/mnist_train_test.cpp](./examples/mnist_train_test.cpp "mnist_train_test.cpp")

## Todo List
* ~~fix train error when batch > 1 issue.~~
* ~~add load & save model function.~~
* add more layer,such as batch normalization layer,dropout layer,etc.
* add weight regular,momentum.
* port to other platforms,such as linux,mac,android,iOS,etc.
* optimize network train/test speed.
* add more optimize method.
* add unit test.
* ~~add license.~~

## Bug Report
* Use [github issues](https://github.com/xylcbd/EasyCNN/issues "issues") please.

## Pull Request
* Pull request is welcome.

## License
This project is released under the [WTFPL LICENSE](http://www.wtfpl.net/ "WTFPL LICENSE").
