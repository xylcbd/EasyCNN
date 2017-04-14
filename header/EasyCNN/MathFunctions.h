#pragma once

namespace EasyCNN
{
	//all of these functions below is run on single thread, maybe they are SIMD optimized.

	void normal_distribution_init(float* data, const size_t size, const float mean_value, const float standard_deviation);
	void uniform_distribution_init(float* data, const size_t size, const float low_value, const float high_deviation);
	void const_distribution_init(float* data, const size_t size, const float const_value);
	void xavier_init(float* data, const size_t size, const size_t fan_in, const size_t fan_out);

	float moving_average(float avg, const int acc_number, float value);

	//c = a*b
	void mul(const float* a, const float* b, float* c, const size_t len);
	//a *= b
	void mul_inplace(float* a, const float* b, const size_t len);

	//a /= b
	void div_inplace(float* a, const float b, const size_t len);

	void sigmoid(const float* x, float* y, const size_t len);	
	void df_sigmoid(const float* x, float* y, const size_t len);

	void tanh(const float* x, float* y, const size_t len);
	void df_tanh(const float* x, float* y, const size_t len);	

	void relu(const float* x, float* y, const size_t len);
	void df_relu(const float* x, float* y, const size_t len);

	//
	void fullconnect(const float* input, const float* weight, const float* bias,float* output,
		const size_t n, const size_t is, const size_t os);

	//mode: 0-validate,1-same
	void convolution2d(const float* input, const float* kernel, const float* bias, float* output,
		const size_t in, const size_t ic, const size_t iw, const size_t ih,
		const size_t kn, const size_t kw, const size_t kh, const size_t kws, const size_t khs,
		const size_t ow, const size_t oh,
		const int mode);
};