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

	const int num = 1;
	const int channels = 1;
	const int width = 32;
	const int height = 32;	

	EasyCNN::NetWork network;
	network.setInputSize(EasyCNN::BucketSize(1,channels, width, height));
	//input data layer 0
	std::shared_ptr<EasyCNN::InputLayer> _0_inputLayer(std::make_shared<EasyCNN::InputLayer>());
	network.addayer(_0_inputLayer);
	//convolution layer 1
	std::shared_ptr<EasyCNN::ConvolutionLayer> _1_convLayer(std::make_shared<EasyCNN::ConvolutionLayer>());	
	_1_convLayer->setParamaters(EasyCNN::BucketSize(6,1,5,5),1,1, true);
	network.addayer(_1_convLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());
	//pooling layer 2
	std::shared_ptr<EasyCNN::PoolingLayer> _2_poolingLayer(std::make_shared<EasyCNN::PoolingLayer>());
	_2_poolingLayer->setParamaters(EasyCNN::PoolingLayer::PoolingType::MaxPooling, EasyCNN::BucketSize(1,6, 2, 2),2,2);
	network.addayer(_2_poolingLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());
	//convolution layer 3
	std::shared_ptr<EasyCNN::ConvolutionLayer> _3_convLayer(std::make_shared<EasyCNN::ConvolutionLayer>());
	_3_convLayer->setParamaters(EasyCNN::BucketSize(16,1,5, 5), 1,1, true);
	network.addayer(_3_convLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());
	//pooling layer 4
	std::shared_ptr<EasyCNN::PoolingLayer> _4_pooingLayer(std::make_shared<EasyCNN::PoolingLayer>());
	_4_pooingLayer->setParamaters(EasyCNN::PoolingLayer::PoolingType::MaxPooling, EasyCNN::BucketSize(1,16, 2, 2),2,2);
	network.addayer(_4_pooingLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());
	//full connect layer 5
	std::shared_ptr<EasyCNN::FullconnectLayer> _5_fullconnectLayer(std::make_shared<EasyCNN::FullconnectLayer>());
	_5_fullconnectLayer->setOutpuBuckerSize(EasyCNN::BucketSize(1,512, 1, 1));
	_5_fullconnectLayer->setParamaters(true);
	network.addayer(_5_fullconnectLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());
	//full connect layer 6
	std::shared_ptr<EasyCNN::FullconnectLayer> _6_fullconnectLayer(std::make_shared<EasyCNN::FullconnectLayer>());
	_6_fullconnectLayer->setOutpuBuckerSize(EasyCNN::BucketSize(1,10, 1, 1));
	_6_fullconnectLayer->setParamaters(true);
	network.addayer(_6_fullconnectLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());
	//soft max layer 6
	std::shared_ptr<EasyCNN::SoftmaxLayer> _7_softmaxLayer(std::make_shared<EasyCNN::SoftmaxLayer>());
	network.addayer(_7_softmaxLayer);

	//train
	int idx = 0;
	std::shared_ptr<EasyCNN::DataBucket> inputDataBucket = std::make_shared<EasyCNN::DataBucket>(EasyCNN::BucketSize(num, channels, width, height));
	while (true)
	{
		std::cout << "\nidx = " << idx++ << std::endl;		
		//TODO : load data
		network.forward(inputDataBucket);
		network.backward();
		if (idx > 10)
		{
			break;
		}
	}
	return 0;
}