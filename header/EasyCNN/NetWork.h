#pragma once
#include <memory>
#include <vector>
#include "EasyCNN/Configure.h"
#include "EasyCNN/Layer.h"

namespace EasyCNN
{
	class NetWork
	{
	public:
		NetWork();
		virtual ~NetWork();
		void addayer(std::shared_ptr<Layer> layer);
	public:
		void setInputSize(const DataSize size);
		void forward(const std::shared_ptr<DataBucket> inputDataBucket);
		void backward(std::shared_ptr<EasyCNN::DataBucket> labelDataBucket);
	private:
		std::vector<std::shared_ptr<Layer>> layers;
		std::vector<std::shared_ptr<DataBucket>> dataBuckets;
	};
}