//configure
#include "EasyCNN/Configure.h"
//layers
#include "EasyCNN/Layer.h"
#include "EasyCNN/ActivationLayer.h"
#include "EasyCNN/InputLayer.h"
#include "EasyCNN/ConvolutionLayer.h"
#include "EasyCNN/PoolingLayer.h"
#include "EasyCNN/FullconnectLayer.h"
#include "EasyCNN/SoftmaxLayer.h"
//network
#include "EasyCNN/NetWork.h"


EasyCNN::NetWork::NetWork()
{
	EASYCNN_LOG_VERBOSE("NetWork constructed.");
}
EasyCNN::NetWork::~NetWork()
{
	EASYCNN_LOG_VERBOSE("NetWork destructed.");
}
void EasyCNN::NetWork::setInputSize(const DataSize size)
{
	EASYCNN_LOG_VERBOSE("NetWork setInputSize begin.");
	easyAssert(size.number > 0 && size.channels > 0 && size.width > 0 && size.height > 0, "parameter invalidate.");
	easyAssert(dataBuckets.empty(), "dataBuckets must be empty now!");
	std::shared_ptr<DataBucket> dataBucket = std::make_shared<DataBucket>(size);
	dataBuckets.push_back(dataBucket);
	EASYCNN_LOG_VERBOSE("NetWork setInputSize end.");
}
void EasyCNN::NetWork::addayer(std::shared_ptr<Layer> layer)
{
	const auto layer_type = layer->getLayerType();
	EASYCNN_LOG_VERBOSE("NetWork addayer begin , type : %s",layer_type.c_str());
	layers.push_back(layer);

	easyAssert(dataBuckets.size() >= 1,"bucket count is less than 1.");
	const std::shared_ptr<DataBucket> prevDataBucket = dataBuckets[dataBuckets.size() - 1];
	easyAssert(prevDataBucket.get() != nullptr,"previous bucket is null.");
	const DataSize inputSize = prevDataBucket->getSize();
	layer->setInputBucketSize(inputSize);	
	layer->solveInnerParams();
	const DataSize outputSize = layer->getOutputBucketSize();
	std::shared_ptr<DataBucket> dataBucket = std::make_shared<DataBucket>(outputSize);
	//dataBucket setting params
	dataBuckets.push_back(dataBucket);
	EASYCNN_LOG_VERBOSE("NetWork addayer end. add data bucket done.");
}
void EasyCNN::NetWork::backward(std::shared_ptr<EasyCNN::DataBucket> labelDataBucket)
{
	EASYCNN_LOG_VERBOSE("NetWork backward begin.");
	easyAssert(layers.size() > 1,"layer count is less than 2.");
	easyAssert(layers[0]->getLayerType() == InputLayer::layerType,"first layer is not input layer.");

	//get ACE error
	const auto lastOutputData = dataBuckets[dataBuckets.size() - 1];
	easyAssert(lastOutputData->getSize() == labelDataBucket->getSize(),"last data bucket's size must be equals with label.");
	const float* labelData = labelDataBucket->getData().get();
	const float* outputData = lastOutputData->getData().get();
	float loss = 0.0f;
	for (size_t i = 0; i < lastOutputData->getSize().totalSize();i++)
	{
		loss += labelData[i] * std::log(outputData[i]);
	}
	loss *= -1.0f;
	EASYCNN_LOG_VERBOSE("loss = %f", loss);

	for (size_t i = 0; i < layers.size();i++)
	{
		EASYCNN_LOG_VERBOSE("NetWork layer[%d](%s) backward begin.",i,layers[i]->getLayerType().c_str());
		layers[i]->backward(dataBuckets[i], dataBuckets[i+1]);
		EASYCNN_LOG_VERBOSE("NetWork layer[%d](%s) backward end.", i, layers[i]->getLayerType().c_str());
	}

	EASYCNN_LOG_VERBOSE("NetWork backward end.");
}
void EasyCNN::NetWork::forward(const std::shared_ptr<DataBucket> inputDataBucket)
{
	EASYCNN_LOG_VERBOSE("NetWork forward end.");
	easyAssert(layers.size() > 1, "layer count is less than 2.");
	easyAssert(layers[0]->getLayerType() == InputLayer::layerType, "first layer is not input layer.");
	easyAssert(dataBuckets.size() > 0, "data buckets is not ready.");
	//copy data from inputDataBucket
	inputDataBucket->cloneTo(*dataBuckets[0]);

	for (size_t i = 0; i < layers.size(); i++)
	{
		EASYCNN_LOG_VERBOSE("NetWork layer[%d](%s) forward begin.", i, layers[i]->getLayerType().c_str());
		layers[i]->forward(dataBuckets[i], dataBuckets[i+1]);
		EASYCNN_LOG_VERBOSE("NetWork layer[%d](%s) forward end.", i, layers[i]->getLayerType().c_str());
	}
	EASYCNN_LOG_VERBOSE("NetWork forward end.");
}