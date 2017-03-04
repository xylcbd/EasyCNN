#include "EasyCNN/Optimizer.h"

namespace EasyCNN
{
	//SGD
	//w -= lr*g
	void SGD::update(std::vector<std::shared_ptr<DataBucket>> params, const std::vector<std::shared_ptr<DataBucket>> gradients)
	{
		easyAssert(params.size() == gradients.size(), "size of param and size of diff must be equals.");
		for (size_t i = 0; i < params.size(); i++)
		{
			easyAssert(params[i]->getSize() == gradients[i]->getSize(), "size of param[i] and size of diff[i] must be equals.");
			float* paramData = params[i]->getData().get();
			const float* gradientData = gradients[i]->getData().get();
			for (size_t j = 0; j < params[i]->getSize().totalSize(); j++)
			{
				paramData[j] -= lr*gradientData[j];
			}
		}
	}

	//SGDWithMomentum
	//prev_m = momentum*prev_m+g[t]
	//w -= lr*prev_m
	void SGDWithMomentum::update(std::vector<std::shared_ptr<DataBucket>> params, const std::vector<std::shared_ptr<DataBucket>> gradients)
	{
		easyAssert(params.size() == gradients.size(), "size of param and size of gradient must be equals.");
		if (prevM.size() != params.size())
		{
			prevM.resize(params.size());
		}
		for (size_t i = 0; i < params.size(); i++)
		{
			if (prevM[i].get() == nullptr || prevM[i]->getSize() != params[i]->getSize())
			{
				prevM[i].reset(new DataBucket(params[i]->getSize()));
				prevM[i]->fillData(0.0f);
			}
			easyAssert(params[i]->getSize() == gradients[i]->getSize(), "size of param[i] and size of gradient[i] must be equals.");
			float* paramData = params[i]->getData().get();
			const float* gradientData = gradients[i]->getData().get();
			float* prevMData = prevM[i]->getData().get();
			for (size_t j = 0; j < params[i]->getSize().totalSize(); j++)
			{
				prevMData[j] = momentum*prevMData[j] + gradientData[j];
				paramData[j] -= lr*prevMData[j];
			}
		}
	}
}//namespace