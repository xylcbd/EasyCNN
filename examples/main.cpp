#include <iostream>
#include <cassert>
#include "EasyCNN/EasyCNN.h"
#include "mnist_data_loader.h"

static bool fetch_data(const std::vector<image_t>& images,std::shared_ptr<EasyCNN::DataBucket> inputDataBucket, 
	const std::vector<label_t>& labels, std::shared_ptr<EasyCNN::DataBucket> labelDataBucket,
	const size_t offset, const size_t length)
{
	assert(images.size() == labels.size() && inputDataBucket->getSize().number == labelDataBucket->getSize().number);
	if (offset >= images.size())
	{
		return false;
	}
	size_t actualEndPos = offset + length;
	if (actualEndPos > images.size())
	{
		//image data
		auto inputDataSize = inputDataBucket->getSize();
		inputDataSize.number = images.size() - offset;
		actualEndPos = offset + inputDataSize.number;
		inputDataBucket.reset(new EasyCNN::DataBucket(inputDataSize));
		//label data
		auto labelDataSize = labelDataBucket->getSize();
		labelDataSize.number = inputDataSize.number;
		labelDataBucket.reset(new EasyCNN::DataBucket(inputDataSize));
	}
	//copy
	const size_t sizePerImage = inputDataBucket->getSize()._3DSize();
	const size_t sizePerLabel = labelDataBucket->getSize()._3DSize();
	assert(sizePerImage == images[0].channels*images[0].width*images[0].height);
	//scale to 0.0f~1.0f
	const float scaleRate = 1.0f / 256.0f;
	for (size_t i = offset; i < actualEndPos; i++)
	{
		//image data
		float* inputData = inputDataBucket->getData().get() + (i - offset)*sizePerImage;
		const uint8_t* imageData = &images[i].data[0];
		for (size_t j = 0; j < sizePerImage;j++)
		{
			inputData[j] = (float)imageData[j] * scaleRate;
		}
		//label data
		float* labelData = labelDataBucket->getData().get() + (i - offset)*sizePerLabel;
		const uint8_t label = labels[i].data;
		for (size_t j = 0; j < sizePerLabel; j++)
		{
			if (j == label)
			{
				labelData[j] = 1.0f;
			}
			else
			{
				labelData[j] = 0.0f;
			}			
		}
	}
	return true;
}
int main(int argc, char* argv[])
{
	const std::string mnist_train_images_file = "../../res/mnist_data/train-images.idx3-ubyte";
	const std::string mnist_train_labels_file = "../../res/mnist_data/train-labels.idx1-ubyte";
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
	//train data & validate data
	//train
	std::vector<image_t> train_images(static_cast<size_t>(images.size()*0.8f));
	std::vector<label_t> train_labels(static_cast<size_t>(labels.size()*0.8f));
	std::copy(images.begin(), images.begin() + train_images.size(), train_images.begin());
	std::copy(labels.begin(), labels.begin() + train_labels.size(), train_labels.begin());
	//validate
	std::vector<image_t> validate_images(images.size() - train_images.size());
	std::vector<label_t> validate_labels(labels.size() - train_labels.size());
	std::copy(images.begin() + train_images.size(), images.end(), validate_images.begin());
	std::copy(labels.begin() + train_labels.size(), labels.end(), validate_labels.begin());

	const size_t epoch = 10;
	const size_t batch = 64;
	const size_t channels = images[0].channels;
	const size_t width = images[0].width;
	const size_t height = images[0].height;

	EasyCNN::NetWork network;
	network.setInputSize(EasyCNN::DataSize(batch,channels, width, height));
	//input data layer 0
	std::shared_ptr<EasyCNN::InputLayer> _0_inputLayer(std::make_shared<EasyCNN::InputLayer>());
	network.addayer(_0_inputLayer);
	//convolution layer 1
	std::shared_ptr<EasyCNN::ConvolutionLayer> _1_convLayer(std::make_shared<EasyCNN::ConvolutionLayer>());	
	_1_convLayer->setParamaters(EasyCNN::ParamSize(6,1,5,5),1,1, true);
	network.addayer(_1_convLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());
	//pooling layer 2
	std::shared_ptr<EasyCNN::PoolingLayer> _2_poolingLayer(std::make_shared<EasyCNN::PoolingLayer>());
	_2_poolingLayer->setParamaters(EasyCNN::PoolingLayer::PoolingType::MaxPooling, EasyCNN::ParamSize(1, 6, 2, 2), 2, 2);
	network.addayer(_2_poolingLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());
	//convolution layer 3
	std::shared_ptr<EasyCNN::ConvolutionLayer> _3_convLayer(std::make_shared<EasyCNN::ConvolutionLayer>());
	_3_convLayer->setParamaters(EasyCNN::ParamSize(16, 6, 5, 5), 1, 1, true);
	network.addayer(_3_convLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());
	//pooling layer 4
	std::shared_ptr<EasyCNN::PoolingLayer> _4_pooingLayer(std::make_shared<EasyCNN::PoolingLayer>());
	_4_pooingLayer->setParamaters(EasyCNN::PoolingLayer::PoolingType::MaxPooling, EasyCNN::ParamSize(1, 16, 2, 2), 2, 2);
	network.addayer(_4_pooingLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());
	//full connect layer 5
	std::shared_ptr<EasyCNN::FullconnectLayer> _5_fullconnectLayer(std::make_shared<EasyCNN::FullconnectLayer>());
	_5_fullconnectLayer->setOutpuBuckerSize(EasyCNN::DataSize(batch, 512, 1, 1));
	_5_fullconnectLayer->setParamaters(true);
	network.addayer(_5_fullconnectLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());
	//full connect layer 6
	std::shared_ptr<EasyCNN::FullconnectLayer> _6_fullconnectLayer(std::make_shared<EasyCNN::FullconnectLayer>());
	_6_fullconnectLayer->setOutpuBuckerSize(EasyCNN::DataSize(batch, 10, 1, 1));
	_6_fullconnectLayer->setParamaters(true);
	network.addayer(_6_fullconnectLayer);
	network.addayer(std::make_shared<EasyCNN::ReluLayer>());
	//soft max layer 6
	std::shared_ptr<EasyCNN::SoftmaxLayer> _7_softmaxLayer(std::make_shared<EasyCNN::SoftmaxLayer>());
	network.addayer(_7_softmaxLayer);

	//train
	std::shared_ptr<EasyCNN::DataBucket> inputDataBucket = std::make_shared<EasyCNN::DataBucket>(EasyCNN::DataSize(batch, channels, width, height));
	std::shared_ptr<EasyCNN::DataBucket> labelDataBucket = std::make_shared<EasyCNN::DataBucket>(EasyCNN::DataSize(batch, 10, 1, 1));
	size_t epochIdx = 0;
	while (epochIdx < epoch)
	{
		size_t offset = 0;
		while (true)
		{
			if (!fetch_data(train_images, inputDataBucket, train_labels, labelDataBucket, offset, batch))
			{
				break;
			}			
			network.forward(inputDataBucket);
			network.backward(labelDataBucket);			
			offset += batch;
		}
		//TODO : test on validate data & get loss / accuracy
		std::cout << "\nepoch : " << epochIdx++ << std::endl;
	}
	return 0;
}