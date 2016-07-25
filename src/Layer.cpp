#include "EasyCNN/Layer.h"

void EasyCNN::Layer::setInputBucketSize(const EasyCNN::DataSize size)
{
	inputSize = size;
}
EasyCNN::DataSize EasyCNN::Layer::getInputBucketSize() const
{
	return inputSize;
}
void EasyCNN::Layer::solveInnerParams()
{
	outputSize = inputSize;
}
void EasyCNN::Layer::setOutpuBuckerSize(const DataSize size)
{
	outputSize = size;
}
EasyCNN::DataSize EasyCNN::Layer::getOutputBucketSize() const
{
	return outputSize;
}