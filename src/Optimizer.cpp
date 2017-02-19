#include "EasyCNN/Optimizer.h"
//SGD
//w -= lr*g
void EasyCNN::SGD::update(std::vector<std::shared_ptr<EasyCNN::DataBucket>> param, const std::vector<std::shared_ptr<EasyCNN::DataBucket>> gradient)
{
	easyAssert(param.size() == gradient.size(), "size of param and size of diff must be equals.");
	for (size_t i = 0; i < param.size();i++)
	{
		easyAssert(param[i]->getSize() == gradient[i]->getSize(), "size of param[i] and size of diff[i] must be equals.");
		float* paramData = param[i]->getData().get();
		const float* gradientData = gradient[i]->getData().get();
		for (size_t j = 0; j < param[i]->getSize().totalSize();j++)
		{
			paramData[j] -= lr*gradientData[j];
		}		
	}
}

//SGDWithMomentum
//prev_m = momentum*prev_m+g[t]
//w -= lr*prev_m
void EasyCNN::SGDWithMomentum::update(std::vector<std::shared_ptr<EasyCNN::DataBucket>> param, const std::vector<std::shared_ptr<EasyCNN::DataBucket>> gradient)
{
	easyAssert(param.size() == gradient.size(), "size of param and size of gradient must be equals.");
	if (prevM.size() != param.size())
	{
		prevM.resize(param.size());
	}
	for (size_t i = 0; i < param.size(); i++)
	{
		if (prevM[i].get() == nullptr || prevM[i]->getSize() != param[i]->getSize())
		{
			prevM[i].reset(new EasyCNN::DataBucket(param[i]->getSize()));
			prevM[i]->fillData(0.0f);
		}
		easyAssert(param[i]->getSize() == gradient[i]->getSize(), "size of param[i] and size of gradient[i] must be equals.");
		float* paramData = param[i]->getData().get();
		const float* gradientData = gradient[i]->getData().get();
		float* prevMData = prevM[i]->getData().get();
		for (size_t j = 0; j < param[i]->getSize().totalSize(); j++)
		{
			prevMData[j] = momentum*prevMData[j] + gradientData[j];
			paramData[j] -= lr*prevMData[j];
		}
	}
}