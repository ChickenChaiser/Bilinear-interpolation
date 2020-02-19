
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "EasyBMP/EasyBMP.h"
#include <iomanip>
#include <math.h>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <ctime>




using namespace std;
//----------------------------------------GPU--------------------------------------------

#define CUDA_CALL(x) do { cudaError_t err = x; if (( err ) != cudaSuccess ) { \
	printf ("Error \"%s\" at %s :%d \n" , cudaGetErrorString(err), \
		__FILE__ , __LINE__ ) ; exit(-1); \
}} while (0)

#define ARR(T, i, j) (T[(i) + (j) * width])

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __RET_PTR   "l"
#else
#define __RET_PTR   "r"
#endif

union RGBApixel_
{
	RGBApixel p;
	int i;
};

static __device__ __inline__ void interpolateGPU(
	const RGBApixel* pixels, RGBApixel* output, int width, float x, float y)
{
	int px = (int)x; // floor of x
	int py = (int)y; // floor of y
	const int stride = width;
	const RGBApixel* p0 = &pixels[0] + px + py * stride; // pointer to first pixel

	// Load the four neighboring pixels
	RGBApixel_ p1_; p1_.i = *(int*)&p0[0 + 0 * stride];
	RGBApixel_ p2_; p2_.i = *(int*)&p0[1 + 0 * stride];
	RGBApixel_ p3_; p3_.i = *(int*)&p0[0 + 1 * stride];
	RGBApixel_ p4_; p4_.i = *(int*)&p0[1 + 1 * stride];

	const RGBApixel& p1 = p1_.p;
	const RGBApixel& p2 = p2_.p;
	const RGBApixel& p3 = p3_.p;
	const RGBApixel& p4 = p4_.p;

	// Calculate the weights for each pixel
	float fx = x - px;
	float fy = y - py;
	float fx1 = 1.0f - fx;
	float fy1 = 1.0f - fy;

	int w1 = fx1 * fy1 * 256.0f + 0.5f;
	int w2 = fx * fy1 * 256.0f + 0.5f;
	int w3 = fx1 * fy * 256.0f + 0.5f;
	int w4 = fx * fy * 256.0f + 0.5f;

	// Calculate the weighted sum of pixels (for each color channel)
	int outr = p1.Red * w1 + p2.Red * w2 + p3.Red * w3 + p4.Red * w4;
	int outg = p1.Green * w1 + p2.Green * w2 + p3.Green * w3 + p4.Green * w4;
	int outb = p1.Blue * w1 + p2.Blue * w2 + p3.Blue * w3 + p4.Blue * w4;
	int outa = p1.Alpha * w1 + p2.Alpha * w2 + p3.Alpha * w3 + p4.Alpha * w4;

	RGBApixel ret;
	ret.Red = (outr + 128) >> 8;
	ret.Green = (outg + 128) >> 8;
	ret.Blue = (outb + 128) >> 8;
	ret.Alpha = (outa + 128) >> 8;

	RGBApixel_* output_ = (RGBApixel_*)output;
	output_->p = ret;
}

__global__ void bilinearGPU(const int width, const int height,
	RGBApixel* input, RGBApixel* output)
{
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (j >= 2 * height) return;
	if (i >= 2 * width) return;

	float x = width * (i - 0.5f) / (float)(2 * width);
	float y = height * (j - 0.5f) / (float)(2 * height);

	interpolateGPU(input, &output[i + j * 2 * width], width, x, y);
}

//----------------------------------------CPU--------------------------------------------


#define ARR(T, i, j) (T[(i) + (j) * width])

static inline void interpolateCPU(const vector<RGBApixel>& pixels, RGBApixel& output, int width, float x, float y)
{
	int px = (int)x; // floor of x
	int py = (int)y; // floor of y
	const int stride = width;
	const RGBApixel* p0 = &pixels[0] + px + py * stride; // pointer to first pixel

	// load the four neighboring pixels
	const RGBApixel& p1 = p0[0 + 0 * stride];
	const RGBApixel& p2 = p0[1 + 0 * stride];
	const RGBApixel& p3 = p0[0 + 1 * stride];
	const RGBApixel& p4 = p0[1 + 1 * stride];

	// Calculate the weights for each pixel
	float fx = x - px;
	float fy = y - py;
	float fx1 = 1.0f - fx;
	float fy1 = 1.0f - fy;

	int w1 = fx1 * fy1 * 256.0f + 0.5f;
	int w2 = fx * fy1 * 256.0f + 0.5f;
	int w3 = fx1 * fy * 256.0f + 0.5f;
	int w4 = fx * fy * 256.0f + 0.5f;

	// Calculate the weighted sum of pixels (for each color channel)
	int outr = p1.Red * w1 + p2.Red * w2 + p3.Red * w3 + p4.Red * w4;
	int outg = p1.Green * w1 + p2.Green * w2 + p3.Green * w3 + p4.Green * w4;
	int outb = p1.Blue * w1 + p2.Blue * w2 + p3.Blue * w3 + p4.Blue * w4;
	int outa = p1.Alpha * w1 + p2.Alpha * w2 + p3.Alpha * w3 + p4.Alpha * w4;

	output.Red = (outr + 128) >> 8;
	output.Green = (outg + 128) >> 8;
	output.Blue = (outb + 128) >> 8;
	output.Alpha = (outa + 128) >> 8;
}

void bilinearCPU(const int width, const int height,
	vector<RGBApixel>& input, vector<RGBApixel>& output)
{
#pragma omp parallel for
	for (int j = 0; j < 2 * height; j++)
		for (int i = 0; i < 2 * width; i++)
		{
			float x = width * (i - 0.5f) / (float)(2 * width);
			float y = height * (j - 0.5f) / (float)(2 * height);

			interpolateCPU(input, output[i + j * 2 * width], width, x, y);
		}
}

int main(int argc, char* argv[])
{
	char* filename = "cat.bmp";

	float timerValueGPU = 0.0f;
	float timerValueCPU = 0.0f;
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	BMP AnImage;
	AnImage.ReadFromFile(filename);
	int width = AnImage.TellWidth();
	int height = AnImage.TellHeight();

	vector<RGBApixel> input(width * (height + 1) + 1);
	vector<RGBApixel> output(4 * width * height);
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
			input[i + j * width] = AnImage.GetPixel(i, j);
	memset(&input[height * width], 0, (width + 1) * sizeof(RGBApixel));

	RGBApixel* dinput, * doutput;
	CUDA_CALL(cudaMalloc(&dinput, sizeof(RGBApixel) * input.size()));
	CUDA_CALL(cudaMalloc(&doutput, sizeof(RGBApixel) * output.size()));
	CUDA_CALL(cudaMemcpy(dinput, &input[0], sizeof(RGBApixel) * input.size(), cudaMemcpyHostToDevice));


	dim3 szblock(128, 1, 1);
	dim3 nblocks(2 * width / szblock.x, 2 * height, 1);
	if (2 * width % szblock.x) nblocks.x++;

	cudaEventRecord(start, 0);

	bilinearGPU << <nblocks, szblock >> > (width, height, dinput, doutput);
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueGPU, start, stop);
	printf("\n GPU calculation time %f msec\n", timerValueGPU);

	struct timespec finish;

	CUDA_CALL(cudaMemcpy(&output[0], doutput, sizeof(RGBApixel) * output.size(), cudaMemcpyDeviceToHost));

	AnImage.SetSize(2 * width, 2 * height);
	for (int i = 0; i < 2 * width; i++)
		for (int j = 0; j < 2 * height; j++)
			AnImage.SetPixel(i, j, output[i + j * 2 * width]);

	AnImage.WriteToFile("output_gpu.bmp");

	CUDA_CALL(cudaFree(dinput));
	CUDA_CALL(cudaFree(doutput));
	

	clock_t startc;
	clock_t stopc;
	startc = clock();
	bilinearCPU(width, height, input, output);
	stopc = clock();
	printf("\n CPU calculation time %f sec\n", ((double)(stopc - startc)) / ((double)CLOCKS_PER_SEC));
	
	//printf("CPU kernel time = %f sec\n", get_time_diff(&start, &finish));

	AnImage.SetSize(2 * width, 2 * height);
	for (int i = 0; i < 2 * width; i++)
		for (int j = 0; j < 2 * height; j++)
			AnImage.SetPixel(i, j, output[i + j * 2 * width]);

	AnImage.WriteToFile("output_cpu.bmp");
 
	return 0;
}
