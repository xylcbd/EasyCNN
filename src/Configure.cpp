#include <algorithm>
#include "EasyCNN/Configure.h"
#include "EasyCNN/EasyAssert.h"

#if WITH_PARALLEL_SUPPORT
namespace EasyCNN
{
	std::shared_ptr<ThreadPool> thread_pool(new ThreadPool(2));

	int get_thread_num()
	{
		return thread_pool->size();
	}
	int set_thread_num(const int num)
	{
		easyAssert(num > 0, "number of thread must be larger than 0");
		if (num != get_thread_num())
		{
			thread_pool.reset(new ThreadPool(num));
		}
		return get_thread_num();
	}
}//namespace
#endif //WITH_PARALLEL_SUPPORT