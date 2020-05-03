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

    int sub_size = (int)ceil(Size*1.0/1024);
    // The resultant number of blocks obtained on dividing the array into blocks of 1024 elements
    int final_size = (int)ceil(sub_size*1.0/1024);
    // The resultant number of blocks obtained on dividing the array into blocks of 2^20 elements

    float* d_array, *d_out, *d_sum;

    cudaMalloc((void**)&d_array, Size*sizeof(float));
    cudaMalloc((void**)&d_out, sub_size*sizeof(float));
    cudaMalloc((void**)&d_sum, final_size*sizeof(float));

    cudaMemcpy(d_array, h_array, sizeof(float) * Size, cudaMemcpyHostToDevice);

    float *h_sum;
    h_sum = (float*)malloc(final_size * sizeof(float)); 

    start = clock();

    Array_Add <<<ceil(Size*1.0/1024), 1024, 1024*sizeof(float)>>> (d_out, d_array, Size);

    Array_Add <<<final_size, 1024, 1024*sizeof(float)>>> (d_sum, d_out, ceil(Size*1.0/1024));

    end = clock();

    cudaMemcpy(h_sum, d_sum, final_size*sizeof(float), cudaMemcpyDeviceToHost);
    float sum = h_sum[0];
    for(int i=1; i<final_size; i++)
        sum += h_sum[i];

    cout << "\nThe time taken by GPU is " << (double)(end-start) << " microseconds\n";

    cudaFree(d_array);
    cudaFree(d_out);
    cudaFree(d_sum);

    return sum;
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
    float *h_array;
    h_array = (float*)malloc(Size*sizeof(float));
    for(int i=0; i<Size; i++)
        h_array[i] = 10;

    float h_sum = Find_Sum_GPU(h_array, Size);

    float naive_sum = Find_Sum_CPU(h_array, Size);

    cout << "\nThe sum is " << h_sum << endl;

    if(h_sum == naive_sum)
        cout << "Result computed correctly.";
    else
        cout << "Result wrong!";
}