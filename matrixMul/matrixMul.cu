#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
//#include<intrin.h>
//#include<stdint.h>

// Работа с GPU
__device__ long double coef(float* matrix, int n, int k, int j)
{
	return matrix[j + n * (k + 1)] / matrix[j * (n + 1)];
}

__global__ void GAUSS(float* matrix, int n)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x + n;
	int k;

	for (int j = 0; j < n - 1; j++)
	{
		k = j;
		while (k < n - 1)
		{// [элемент]-=[коэффициент деления элемента на исходный этого столбца]*[исходный]
			matrix[tid + n * k + j] -= coef(matrix, n, k, j) * matrix[tid + n * k + j - n * (1 + k - j)];
			k++;//сдвигаемся вниз
		}
	}
}

//определитель для CPU
void determinant(float* A, int n)
{
	int i, j, k;
	double coef, det = 1;
	for (k = 0; k < n; k++)
	{
		for (i = k + 1; i < n; i++)
		{
			coef = A[i * n + k] / A[k * n + k];
			for (j = 0; j < n; j++)
				A[i * n + j] -= A[k * n + j] * coef;
		}
	}
}

void main()
{
	srand(time(NULL));
	int n = 0;
	printf("Matrix nxn; n = ");
	scanf("%d", &n);
	float gpuTime = 0;
	double start1;
	// ОП для CPU
	float* HostMatrix = (float*)malloc(n * n * sizeof(float));
	float* cpuMatrix = (float*)malloc(n * n * sizeof(float));

	// Инициализация исходных данных
	printf("Initial Matrix:");
	for (int i = 0; i < n * n; i++)
	{
		HostMatrix[i] = rand() % 8;
		//if (i % n == 0) printf("\n");
		//printf("%g ", HostMatrix[i]);
	}
	printf("\n");
	memcpy(cpuMatrix, HostMatrix, n * n * sizeof(float));

	float* DeviceMatrix = NULL;
	cudaMalloc((void**)&DeviceMatrix, n * n * sizeof(float));

	// Копирование исходных данных в GPU для обработки
	cudaMemcpy(DeviceMatrix, HostMatrix, n * n * sizeof(float),
		cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Запуск ядра

	if (n<1024) GAUSS << < dim3(1,1,1), dim3(n,1,1) >> > (DeviceMatrix, n);
	else GAUSS << < dim3(1024/n+1,1,1), dim3(1024,1,1) >> > (DeviceMatrix, n);

	cudaThreadSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	// Считывание результата из GPU
	cudaMemcpy(HostMatrix, DeviceMatrix, n * n * sizeof(float),
		cudaMemcpyDeviceToHost);

	// Преобразованая матрица
	printf("\n");
	printf("Gauss matrix:");
	for (int i = 0; i < n * n; i++)
	{
		//if (i % n == 0) printf("\n");
		//printf("%f ", HostMatrix[i]);
	}
	printf("\n\n");

	printf("GPU time: %g milliseconds\n", gpuTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	long double det = 1;
	for (int i = 0; i < n * n; i += (n + 1)) det *= HostMatrix[i];
	printf("Determinant by GPU = %g ", det);
	printf("\n\n");

	//Работа CPU
	det = 1;
	start1 = (float)clock() / CLOCKS_PER_SEC;
	determinant(cpuMatrix, n);
	start1 = (float)clock() / CLOCKS_PER_SEC - start1;
	printf("CPU time: %g milliseconds", start1 * 1000);
	for (int i = 0; i < n; i++)
	{
		det *= cpuMatrix[i * n + i];
	}
	printf("\nDeterminant by CPU = %g \n", det);

	//Освобождение памяти
	cudaFree(DeviceMatrix);
	free(HostMatrix);
	free(cpuMatrix);
}
