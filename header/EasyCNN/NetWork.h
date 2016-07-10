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
		void forward(std::shared_ptr<DataBucket> inputDataBucket);
		void backward();		
	private:
		std::vector<std::shared_ptr<Layer>> layers;
		std::vector<std::shared_ptr<DataBucket>> dataBuckets;
	};
}