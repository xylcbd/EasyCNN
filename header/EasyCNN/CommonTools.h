#pragma once
#include <random>
#include "EasyCNN/Configure.h"

namespace EasyCNN
{
	inline void normal_distribution_init(float* data, const size_t size, const float mean_value, const float standard_deviation)
	{
		std::random_device rd;
		std::mt19937 engine(rd());
		std::normal_distribution<float> dist(mean_value,standard_deviation);
		for (size_t i = 0; i < size;i++)
		{
			//for overflow
			data[i] = dist(engine) / 10.0f;
		}
	}
	inline void const_distribution_init(float* data, const size_t size, const float const_value)
	{
		for (size_t i = 0; i < size; i++)
		{
			data[i] = const_value;
		}
	}
}