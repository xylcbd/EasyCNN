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
	train();
	test();

	return 0;
}