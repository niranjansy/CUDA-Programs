#include<iostream>
using namespace std;

#include <time.h>

__global__ void Array_Add(float* d_out, float* d_array, float Size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    extern __shared__ float sh_array[];
    
    if(id < Size)
        sh_array[tid] = d_array[id];
    __syncthreads();

    for(int s = 512; s>0; s = s/2)
    {
        __syncthreads();
        if(id>=Size || id+s>=Size)
            continue;
        if(tid<s)
            sh_array[tid] += sh_array[tid + s];
    }
    if(tid==0)
        d_out[bid] = sh_array[tid];  

    __syncthreads();
}

float Find_Sum_GPU(float h_array[], int Size)
{
    clock_t start, end;

    float* d_array, *d_out, *d_sum;

    cudaMalloc((void**)&d_array, Size*sizeof(float));
    cudaMalloc((void**)&d_out, ceil(Size*1.0/1024)*sizeof(float));
    cudaMalloc((void**)&d_sum, sizeof(float));

    cudaMemcpy(d_array, h_array, sizeof(float) * Size, cudaMemcpyHostToDevice);

    float h_sum;

    start = clock();

    Array_Add <<<ceil(Size*1.0/1024), 1024, 1024*sizeof(float)>>> (d_out, d_array, Size);

    Array_Add <<<1, 1024, 1024*sizeof(float)>>> (d_sum, d_out, ceil(Size*1.0/1024));

    end = clock();

    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cout << "\nThe time taken by GPU is " << (double)(end-start) << " microseconds\n";

    cudaFree(d_Array);
    cudaFree(d_out);
    cudaFree(d_sum);

    return h_sum;
}

float Find_Sum_CPU(float h_array[], int Size)
{
    clock_t tim;
	tim = clock();
    float naive_sum = 0;
    for(int i=0; i<Size; i++)
        naive_sum = naive_sum + h_array[i]; 
    tim = clock() - tim;
    cout << "\nThe time taken by CPU is " << (double)(tim) << " microseconds\n";

    return naive_sum;
}

int main()
{

    int Size;
    cout << "\nEnter the size of the array : ";
    cin >> Size;
    float h_array[Size];
    for(int i=0; i<Size; i++)
        h_array[i] = 10;

    float h_sum = Find_Sum_GPU(h_array, Size);

    float naive_sum = Find_Sum_CPU(h_array, Size);

    cout << "\nThe sum is " << h_sum << endl;
}