#include <string>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

class ub_image
{
public:
	unsigned int w;
	unsigned int h;
	unsigned char *data;

	ub_image ()
	{
		w = 0;
		h = 0;
		data = 0;
	}

	ub_image (unsigned int W, unsigned int H)
	{
		w = W;
		h = H;
		data = ((unsigned char*)malloc(w * h * 4 * sizeof(unsigned char)));
	}

	~ub_image ()
	{
		data = 0;
	}
};

class ub_image_cuda
{
public:
	unsigned int w;
	unsigned int h;
	unsigned char *data;

	ub_image_cuda ()
	{
		w = 0;
		h = 0;
		data = 0;
	}

	ub_image_cuda (unsigned int W, unsigned int H)
	{
		w = W;
		h = H;
		cudaMalloc(((void**)&data), w * h * 4 * sizeof(unsigned char));
	}

	~ub_image_cuda ()
	{
		data = 0;
	}
};

static void upload_to_cuda(const ub_image &src, ub_image_cuda &dst)
{
	if ((dst.w != src.w) || (dst.h != src.h)) {
		cudaFree(dst.data);
		dst.data = 0;
	}
	if (dst.data == 0) {
		dst.w = src.w;
		dst.h = src.h;
		cudaMalloc(((void**)&dst.data), src.w * src.h * 4 * sizeof(unsigned char));
	}
	cudaMemcpy(dst.data, src.data, src.w * src.h * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);
}

static void download_from_cuda(const ub_image_cuda &src, ub_image &dst)
{
	if ((dst.w != src.w) || (dst.h != src.h)) {
		free(dst.data);
		dst.data = 0;
	}
	if (dst.data == 0) {
		dst.w = src.w;
		dst.h = src.h;
		dst.data = ((unsigned char*)malloc(src.w * src.h * 4 * sizeof(unsigned char)));
	}
	cudaMemcpy(dst.data, src.data, src.w * src.h * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}

ub_image load_ub_image(const std::string  &filename)
;

void store_ub_image(const ub_image &image, const std::string  &filename)
;
