#pragma once

namespace EasyCNN
{
	//all of these functions below is run on single thread, maybe they are SIMD optimized.

	void normal_distribution_init(float* data, const size_t size, const float mean_value, const float standard_deviation);
	void const_distribution_init(float* data, const size_t size, const float const_value);

	void mul(const float* a, const float* b, float* c, const size_t len);
	void mul_inplace(float* a_inplace, const float* b, const size_t len);

	void sigmoid(const float* x, float* y, const size_t len);	
	void df_sigmoid(const float* x, float* y, const size_t len);

	void tanh(const float* x, float* y, const size_t len);
	void df_tanh(const float* x, float* y, const size_t len);	

	void relu(const float* x, float* y, const size_t len);
	void df_relu(const float* x, float* y, const size_t len);
};