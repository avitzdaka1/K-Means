#include "main_header.h"
point_t* GlobalPointsCuda = 0;

//	Kernel point relocation function
__global__ void pointRelocationKernel(point_t* pointsArrCuda, double timeInterval, double currentT, int numOfPoints, double cosT, double sinT) {	//	Kernel point relocation function using GPU.
			int id = blockIdx.x * blockDim.x + threadIdx.x;
			if (id < numOfPoints) {
				double centerX = pointsArrCuda[id].a, centerY = pointsArrCuda[id].b;		//	Saves current X, Y coordinates.
				pointsArrCuda[id].x = centerX + (pointsArrCuda[id].radius * cosT);		//	Calculates new X coordinate and stores it in the point's X value.
				pointsArrCuda[id].y = centerY + (pointsArrCuda[id].radius * sinT);		//	Calculates new Y coordinate and stores it in the point's Y value.
			}
}

//	Point relocation function using GPU.
cudaError_t pointRelocationCuda(point_t* pointsArr, double timeInterval,
	double currentT, int numOfPoints, double cosT, double sinT) {
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
	int threadsPerBlock = deviceProp.maxThreadsPerBlock / 3;
	int numOfBlocks = (numOfPoints / threadsPerBlock) + 1;
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) 
		fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
    
	// Launch a kernel on the GPU with one thread for each element.
	pointRelocationKernel<<<numOfBlocks, threadsPerBlock>>>(GlobalPointsCuda, timeInterval, currentT, numOfPoints, cosT, sinT);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
  
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(pointsArr, GlobalPointsCuda, numOfPoints * sizeof(point_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMemcpy failed!");

    return cudaStatus;
}

//	Allocates memory space for the points array in the GPU and copies the points array to that memory.
cudaError_t allocatePointsCuda(point_t* pointsArr, int numOfPoints) {
	cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&GlobalPointsCuda, numOfPoints * sizeof(point_t));
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed!");


    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(GlobalPointsCuda, pointsArr, numOfPoints * sizeof(point_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
        fprintf(stderr, "cudaMemcpy failed!");
    return cudaStatus;
}
