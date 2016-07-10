#include <iostream>
#include <cassert>
#include "EasyCNN/EasyCNN.h"
#include "mnist_data_loader.h"

static void train()
{
	//train
	const std::string mnist_train_images_file = "../res/mnist_data/train-images.idx3-ubyte";
	const std::string mnist_train_labels_file = "../res/mnist_data/train-labels.idx1-ubyte";
	bool success = false;
	//load train images
	std::vector<image_t> images;
	success = load_mnist_images(mnist_train_images_file, images);
	assert(success && images.size() > 0);
	//load train labels
	std::vector<label_t> labels;
	success = load_mnist_labels(mnist_train_labels_file, labels);
	assert(success && labels.size() > 0);
	assert(images.size() == labels.size());
	//TODO
}
static void test()
{
	//test
	const std::string mnist_test_images_file = "../res/mnist_data/t10k-images.idx3-ubyte";
	const std::string mnist_test_labels_file = "../res/mnist_data/t10k-labels.idx1-ubyte";
	bool success = false;
	//load train images
	std::vector<image_t> images;
	success = load_mnist_images(mnist_test_images_file, images);
	assert(success && images.size() > 0);
	//load train labels
	std::vector<label_t> labels;
	success = load_mnist_labels(mnist_test_labels_file, labels);
	assert(success && labels.size() > 0);
	assert(images.size() == labels.size());
	//TODO
}

int main(int argc, char* argv[])
{
// 	train();
// 	test();

	EasyCNN::NetWork network;
	//input data layer 0
	std::shared_ptr<EasyCNN::InputLayer> _0_inputLayer(std::make_shared<EasyCNN::InputLayer>());
	network.addayer(_0_inputLayer);
	//convolution layer 1
	std::shared_ptr<EasyCNN::ConvolutionLayer> _1_convLayer(std::make_shared<EasyCNN::ConvolutionLayer>());	
	network.addayer(_1_convLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());
	//pooling layer 2
	std::shared_ptr<EasyCNN::PoolingLayer> _2_poolingLayer(std::make_shared<EasyCNN::PoolingLayer>());
	network.addayer(_2_poolingLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());
	//convolution layer 3
	std::shared_ptr<EasyCNN::ConvolutionLayer> _3_convLayer(std::make_shared<EasyCNN::ConvolutionLayer>());
	network.addayer(_3_convLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());
	//pooling layer 4
	std::shared_ptr<EasyCNN::PoolingLayer> _4_pooingLayer(std::make_shared<EasyCNN::PoolingLayer>());
	network.addayer(_4_pooingLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());
	//full connect layer 5
	std::shared_ptr<EasyCNN::FullconnectLayer> _5_fullconnectLayer(std::make_shared<EasyCNN::FullconnectLayer>());
	network.addayer(_5_fullconnectLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());
	//soft max layer 6
	std::shared_ptr<EasyCNN::SoftmaxLayer> _6_softmaxLayer(std::make_shared<EasyCNN::SoftmaxLayer>());
	network.addayer(_6_softmaxLayer);

	//train
	int idx = 0;
	while (true)
	{
		std::cout << "\nidx = " << idx++ << std::endl;
		std::shared_ptr<EasyCNN::DataBucket> inputDataBucket;
		//TODO : load data
		network.forward(inputDataBucket);
		network.backward();
	}
	return 0;
}