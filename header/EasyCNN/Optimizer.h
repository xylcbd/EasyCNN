#pragma once
#include <vector>
#include "EasyCNN/Configure.h"
#include "EasyCNN/DataBucket.h"
#include "EasyCNN/ParamBucket.h"

namespace EasyCNN
{
	class Optimizer
	{
	public:
		Optimizer(const float _lr) :lr(_lr){}
		void setLearningRate(const float _lr) { lr = _lr; };
		virtual void update(std::vector<std::shared_ptr<DataBucket>> param, const std::vector<std::shared_ptr<DataBucket>> gradient) = 0;		
	protected:
		float lr = 0.0f;
	};

	class SGD : public Optimizer
	{
	public:
		SGD(const float _lr) :Optimizer(_lr){}
		virtual void update(std::vector<std::shared_ptr<DataBucket>> param, const std::vector<std::shared_ptr<DataBucket>> gradient) override;
	};

	class SGDWithMomentum : public Optimizer
	{
	public:
		SGDWithMomentum(const float _lr, const float _momentum) :Optimizer(_lr), momentum(_momentum){}
		virtual void update(std::vector<std::shared_ptr<DataBucket>> param, const std::vector<std::shared_ptr<DataBucket>> gradient) override;
	private:
		float momentum = 0.0f;
		std::vector<std::shared_ptr<DataBucket>> prevM;
	};
}