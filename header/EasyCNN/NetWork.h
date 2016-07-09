#pragma once
#include "EasyCNN/Configure.h"
#include "EasyCNN/Layer.h"
#include <memory>

namespace EasyCNN
{
	class NetWork
	{
	public:
		NetWork();
		virtual ~NetWork();
		void addayer(std::shared_ptr<Layer> layer);
		void backward();
		void forward();
	private:
	};
}