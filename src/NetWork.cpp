#include <algorithm>
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
float EasyCNN::NetWork::backward(const std::shared_ptr<EasyCNN::DataBucket> labelDataBucket,const float learningRate)
{
	EASYCNN_LOG_VERBOSE("NetWork backward begin.");
	easyAssert(layers.size() > 1,"layer count is less than 2.");
	easyAssert(layers[0]->getLayerType() == InputLayer::layerType,"first layer is not input layer.");

	//get ACE error
	const auto lastOutputData = dataBuckets[dataBuckets.size() - 1];
	const auto lastOutputSize = lastOutputData->getSize();
	easyAssert(lastOutputData->getSize() == labelDataBucket->getSize(),"last data bucket's size must be equals with label.");
	const float* labelData = labelDataBucket->getData().get();
	const float* outputData = lastOutputData->getData().get();
	float loss = 0.0f;
	for (size_t i = 0; i < lastOutputSize._4DSize(); i++)
	{
		loss -= labelData[i] * std::log(outputData[i]) / lastOutputSize.number;
	}

	//////////////////////////////////////////////////////////////////////////
	//label layer backward
	const ParamSize nextDiffSize = ParamSize(1,lastOutputSize.channels, lastOutputSize.height, lastOutputSize.width);
	std::shared_ptr<ParamBucket>& nextDiffBucket(std::make_shared<ParamBucket>(nextDiffSize));
	nextDiffBucket->fillData(0.0f);
	float* nextDiff = nextDiffBucket->getData().get();
	for (size_t on = 0; on < lastOutputSize.number; on++)
	{
		for (size_t nextDiffIdx = 0; nextDiffIdx < nextDiffSize._3DSize(); nextDiffIdx++)
		{
			const size_t dataIdx = on*lastOutputSize._3DSize() + nextDiffIdx;
			nextDiff[nextDiffIdx] -= ((labelData[dataIdx] / (outputData[dataIdx]))) / lastOutputSize.number;
		}
	}

	//other layer backward
	for (int i = (int)(layers.size())-1; i >= 0; i--)
	{
		EASYCNN_LOG_VERBOSE("NetWork layer[%d](%s) backward begin.",i,layers[i]->getLayerType().c_str());
		layers[i]->setLearningRate(learningRate);
		layers[i]->backward(dataBuckets[i], dataBuckets[i + 1], nextDiffBucket);
		EASYCNN_LOG_VERBOSE("NetWork layer[%d](%s) backward end.", i, layers[i]->getLayerType().c_str());
	}
	EASYCNN_LOG_VERBOSE("NetWork backward end.");

	return loss;
}
std::shared_ptr<EasyCNN::DataBucket> EasyCNN::NetWork::forward(const std::shared_ptr<DataBucket> inputDataBucket)
{
	EASYCNN_LOG_VERBOSE("NetWork forward begin.");
	easyAssert(layers.size() > 1, "layer count is less than 2.");
	easyAssert(layers[0]->getLayerType() == InputLayer::layerType, "first layer is not input layer.");
	easyAssert(dataBuckets.size() > 0, "data buckets is not ready.");
	//copy data from inputDataBucket
	//reshape data bucket
	const auto oldNumber = dataBuckets[0]->getSize().number;
	const auto newNumber = inputDataBucket->getSize().number;
	if (newNumber != oldNumber)
	{
		for (size_t i = 0; i < dataBuckets.size(); i++)
		{
			auto newSize = dataBuckets[i]->getSize();
			newSize.number = newNumber;
			dataBuckets[i].reset(new DataBucket(newSize));
		}
	}
	inputDataBucket->cloneTo(*dataBuckets[0]);

	for (size_t i = 0; i < layers.size(); i++)
	{
		EASYCNN_LOG_VERBOSE("NetWork layer[%d](%s) forward begin.", i, layers[i]->getLayerType().c_str());
		layers[i]->forward(dataBuckets[i], dataBuckets[i+1]);
		EASYCNN_LOG_VERBOSE("NetWork layer[%d](%s) forward end.", i, layers[i]->getLayerType().c_str());
	}

	EASYCNN_LOG_VERBOSE("NetWork forward end.");
	return dataBuckets[dataBuckets.size() - 1];
}