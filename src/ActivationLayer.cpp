#include <algorithm>
#include "EasyCNN/ActivationLayer.h"
#include "EasyCNN/MathFunctions.h"

namespace EasyCNN
{
	SigmodLayer::SigmodLayer()
	{

	}
	SigmodLayer::~SigmodLayer()
	{

	}
	DEFINE_LAYER_TYPE(SigmodLayer, "SigmodLayer");
	std::string SigmodLayer::getLayerType() const
	{
		return layerType;
	}
	void SigmodLayer::forward(const std::shared_ptr<DataBucket> prev, std::shared_ptr<DataBucket> next)
	{
		const DataSize prevSize = prev->getSize();
		const DataSize nextSize = next->getSize();

		const float* prevData = prev->getData().get();
		float* nextData = next->getData().get();
#if WITH_PARALLEL_SUPPORT
		if (thread_pool->size() <= 1 || prevSize.totalSize() <= 4 * 1024)
		{
			sigmoid(prevData, nextData, nextSize.totalSize());
		}
		else
		{
			easyAssert(thread_pool->size() > 1, "thread must be larger than 1");
			size_t payload_per_thread = prevSize.totalSize() / thread_pool->size();
			std::vector<std::future<void>> futures;
			for (size_t i = 0; i < (size_t)thread_pool->size(); i++)
			{
				const size_t start = i*payload_per_thread;
				const size_t stop = std::min((i + 1)*payload_per_thread, prevSize.totalSize());
				futures.push_back(thread_pool->enqueue(sigmoid, prevData + start, nextData + start, stop - start));
				if (stop >= prevSize.totalSize())
				{
					break;
				}
			}
			for (size_t i = 0; i < futures.size(); i++)
			{
				futures[i].wait();
			}
		}
#else
		sigmoid(prevData, nextData, nextSize.totalSize());
#endif //WITH_PARALLEL_SUPPORT
	}
	void SigmodLayer::backward(std::shared_ptr<DataBucket> prev, const std::shared_ptr<DataBucket> next,
		std::shared_ptr<DataBucket>& prevDiff, const std::shared_ptr<DataBucket>& nextDiff)
	{
		easyAssert(getPhase() == Phase::Train, "backward only in train phase.")
		const DataSize prevSize = prev->getSize();
		const DataSize nextSize = next->getSize();
		const DataSize prevDiffSize = prevDiff->getSize();
		const DataSize nextDiffSize = nextDiff->getSize();
		const float* prevData = prev->getData().get();
		const float* nextData = next->getData().get();
		float* prevDiffData = prevDiff->getData().get();
		const float* nextDiffData = nextDiff->getData().get();
		easyAssert(prevSize == nextSize, "size must be equal!");
		easyAssert(prevDiffSize == prevSize, "size of prevDiff and size of prev must be equals");

		//update prevDiff
		prevDiff->fillData(0.0f);
#if WITH_PARALLEL_SUPPORT
		if (thread_pool->size() <= 1 || prevSize.totalSize() <= 4 * 1024)
		{
			//calculate current inner diff
			df_sigmoid(nextData, prevDiffData, prevDiffSize.totalSize());
			//multiply next diff
			mul_inplace(prevDiffData, nextDiffData, prevDiffSize.totalSize());
		}
		else
		{
			easyAssert(thread_pool->size() > 1, "thread must be larger than 1");
			auto worker_func = [](const float* nextData, const float* nextDiff, float* prevDiffData, const size_t len){
				//calculate current inner diff
				df_sigmoid(nextData, prevDiffData, len);
				//multiply next diff
				mul_inplace(prevDiffData, nextDiff, len);
			};
			size_t payload_per_thread = prevSize.totalSize() / thread_pool->size();
			std::vector<std::future<void>> futures;
			for (size_t i = 0; i < (size_t)thread_pool->size(); i++)
			{
				const size_t start = i*payload_per_thread;
				const size_t stop = std::min((i + 1)*payload_per_thread, prevSize.totalSize());
				futures.push_back(thread_pool->enqueue(worker_func, nextData + start, nextDiffData + start, prevDiffData + start, stop - start));
				if (stop >= prevSize.totalSize())
				{
					break;
				}
			}
			for (size_t i = 0; i < futures.size(); i++)
			{
				futures[i].wait();
			}
		}
#else
		//calculate current inner diff
		df_sigmoid(nextData, prevDiffData, prevDiffSize.totalSize());
		//multiply next diff
		mul_inplace(prevDiffData, nextDiffData, prevDiffSize.totalSize());
#endif //WITH_PARALLEL_SUPPORT

		//update this layer's param
		//Tanh layer : nop
	}

	TanhLayer::TanhLayer()
	{

	}
	TanhLayer::~TanhLayer()
	{

	}
	DEFINE_LAYER_TYPE(TanhLayer, "TanhLayer");
	std::string TanhLayer::getLayerType() const
	{
		return layerType;
	}


	void TanhLayer::forward(const std::shared_ptr<DataBucket> prev, std::shared_ptr<DataBucket> next)
	{
		const DataSize prevSize = prev->getSize();
		const DataSize nextSize = next->getSize();

		const float* prevData = prev->getData().get();
		float* nextData = next->getData().get();
#if WITH_PARALLEL_SUPPORT
		if (thread_pool->size() <= 1 || prevSize.totalSize() <= 4 * 1024)
		{
			tanh(prevData, nextData, nextSize.totalSize());
		}
		else
		{
			easyAssert(thread_pool->size() > 1, "thread must be larger than 1");
			size_t payload_per_thread = prevSize.totalSize() / thread_pool->size();
			std::vector<std::future<void>> futures;
			for (size_t i = 0; i < (size_t)thread_pool->size(); i++)
			{
				const size_t start = i*payload_per_thread;
				const size_t stop = std::min((i + 1)*payload_per_thread, prevSize.totalSize());
				futures.push_back(thread_pool->enqueue(tanh, prevData + start, nextData + start, stop - start));
				if (stop >= prevSize.totalSize())
				{
					break;
				}
			}
			for (size_t i = 0; i < futures.size(); i++)
			{
				futures[i].wait();
			}
		}
#else
		tanh(prevData, nextData, nextSize.totalSize());
#endif //WITH_PARALLEL_SUPPORT
	}
	void TanhLayer::backward(std::shared_ptr<DataBucket> prev, const std::shared_ptr<DataBucket> next,
		std::shared_ptr<DataBucket>& prevDiff, const std::shared_ptr<DataBucket>& nextDiff)
	{
		easyAssert(getPhase() == Phase::Train, "backward only in train phase.")
			const DataSize prevSize = prev->getSize();
		const DataSize nextSize = next->getSize();
		const DataSize prevDiffSize = prevDiff->getSize();
		const DataSize nextDiffSize = nextDiff->getSize();
		const float* prevData = prev->getData().get();
		const float* nextData = next->getData().get();
		float* prevDiffData = prevDiff->getData().get();
		const float* nextDiffData = nextDiff->getData().get();
		easyAssert(prevSize == nextSize, "size must be equal!");
		easyAssert(prevDiffSize == prevSize, "size of prevDiff and size of prev must be equals");

		//update prevDiff
		prevDiff->fillData(0.0f);		
#if WITH_PARALLEL_SUPPORT
		if (thread_pool->size() <= 1 || prevSize.totalSize() <= 4 * 1024)
		{
			//calculate current inner diff
			df_tanh(nextData, prevDiffData, prevDiffSize.totalSize());
			//multiply next diff
			mul_inplace(prevDiffData, nextDiffData, prevDiffSize.totalSize());
		}
		else
		{
			easyAssert(thread_pool->size() > 1, "thread must be larger than 1");
			auto worker_func = [](const float* nextData, const float* nextDiffData, float* prevDiffData, const size_t len){
				//calculate current inner diff
				df_tanh(nextData, prevDiffData, len);
				//multiply next diff
				mul_inplace(prevDiffData, nextDiffData, len);
			};
			size_t payload_per_thread = prevSize.totalSize() / thread_pool->size();
			std::vector<std::future<void>> futures;
			for (size_t i = 0; i < (size_t)thread_pool->size(); i++)
			{
				const size_t start = i*payload_per_thread;
				const size_t stop = std::min((i + 1)*payload_per_thread, prevSize.totalSize());
				futures.push_back(thread_pool->enqueue(worker_func, nextData + start, nextDiffData + start, prevDiffData + start, stop - start));
				if (stop >= prevSize.totalSize())
				{
					break;
				}
			}
			for (size_t i = 0; i < futures.size(); i++)
			{
				futures[i].wait();
			}
		}
#else
		//calculate current inner diff
		df_tanh(nextData, prevDiffData, prevDiffSize.totalSize());
		//multiply next diff
		mul_inplace(prevDiffData, nextDiffData, prevDiffSize.totalSize());
#endif //WITH_PARALLEL_SUPPORT

		//update this layer's param
		//Tanh layer : nop
	}

	ReluLayer::ReluLayer()
	{

	}
	ReluLayer::~ReluLayer()
	{

	}
	DEFINE_LAYER_TYPE(ReluLayer, "ReluLayer");
	std::string ReluLayer::getLayerType() const
	{
		return layerType;
	}


	void ReluLayer::forward(const std::shared_ptr<DataBucket> prev, std::shared_ptr<DataBucket> next)
	{
		const DataSize prevSize = prev->getSize();
		const DataSize nextSize = next->getSize();

		const float* prevData = prev->getData().get();
		float* nextData = next->getData().get();
#if WITH_PARALLEL_SUPPORT
		if (thread_pool->size() <= 1 || prevSize.totalSize() <= 4 * 1024)
		{
			relu(prevData, nextData, nextSize.totalSize());
		}
		else
		{
			easyAssert(thread_pool->size() > 1, "thread must be larger than 1");
			size_t payload_per_thread = prevSize.totalSize() / thread_pool->size();
			std::vector<std::future<void>> futures;
			for (size_t i = 0; i < (size_t)thread_pool->size(); i++)
			{
				const size_t start = i*payload_per_thread;
				const size_t stop = std::min((i + 1)*payload_per_thread, prevSize.totalSize());
				futures.push_back(thread_pool->enqueue(relu, prevData + start, nextData + start, stop - start));
				if (stop >= prevSize.totalSize())
				{
					break;
				}
			}
			for (size_t i = 0; i < futures.size(); i++)
			{
				futures[i].wait();
			}
		}
#else
		relu(prevData, nextData, nextSize.totalSize());
#endif //WITH_PARALLEL_SUPPORT
	}
	void ReluLayer::backward(std::shared_ptr<DataBucket> prev, const std::shared_ptr<DataBucket> next,
		std::shared_ptr<DataBucket>& prevDiff, const std::shared_ptr<DataBucket>& nextDiff)
	{
		easyAssert(getPhase() == Phase::Train, "backward only in train phase.")
		const DataSize prevSize = prev->getSize();
		const DataSize nextSize = next->getSize();
		const DataSize prevDiffSize = prevDiff->getSize();
		const DataSize nextDiffSize = nextDiff->getSize();
		const float* prevData = prev->getData().get();
		const float* nextData = next->getData().get();
		float* prevDiffData = prevDiff->getData().get();
		const float* nextDiffData = nextDiff->getData().get();
		easyAssert(prevSize == nextSize, "size must be equal!");
		easyAssert(prevDiffSize == prevSize, "size of prevDiff and size of prev must be equals");

		//update prevDiff		
		prevDiff->fillData(0.0f);		
#if WITH_PARALLEL_SUPPORT
		if (thread_pool->size() <= 1 || prevSize.totalSize() <= 4 * 1024)
		{
			//calculate current inner diff
			df_relu(nextData, prevDiffData, prevDiffSize.totalSize());
			//multiply next diff
			mul_inplace(prevDiffData, nextDiffData, prevDiffSize.totalSize());
		}
		else
		{
			easyAssert(thread_pool->size() > 1, "thread must be larger than 1");
			auto worker_func = [](const float* nextData, const float* nextDiffData, float* prevDiffData, const size_t len){
				//calculate current inner diff
				df_relu(nextData, prevDiffData, len);
				//multiply next diff
				mul_inplace(prevDiffData, nextDiffData, len);
			};
			size_t payload_per_thread = prevSize.totalSize() / thread_pool->size();
			std::vector<std::future<void>> futures;
			for (size_t i = 0; i < (size_t)thread_pool->size(); i++)
			{
				const size_t start = i*payload_per_thread;
				const size_t stop = std::min((i + 1)*payload_per_thread, prevSize.totalSize());
				futures.push_back(thread_pool->enqueue(worker_func, nextData + start, nextDiffData + start, prevDiffData + start, stop - start));
				if (stop >= prevSize.totalSize())
				{
					break;
				}
			}
			for (size_t i = 0; i < futures.size(); i++)
			{
				futures[i].wait();
			}
		}
#else
		//calculate current inner diff
		df_relu(nextData, prevDiffData, prevDiffSize.totalSize());
		//multiply next diff
		mul_inplace(prevDiffData, nextDiffData, prevDiffSize.totalSize());
#endif //WITH_PARALLEL_SUPPORT

		//update this layer's param
		//RELU layer : nop
	}
}//namespace