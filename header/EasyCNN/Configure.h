#pragma once

#define WITH_OPENCV_DEBUG 0
#define WITH_PARALLEL_SUPPORT 1

#if WITH_PARALLEL_SUPPORT
#include "EasyCNN/ThreadPool.h"

namespace EasyCNN
{
	//thread pool obj
	extern std::shared_ptr<ThreadPool> thread_pool;

	//get thread number
	int get_thread_num();
	//set thread number, and returned support thread number
	int set_thread_num(const int num);
};
#endif //WITH_PARALLEL_SUPPORT