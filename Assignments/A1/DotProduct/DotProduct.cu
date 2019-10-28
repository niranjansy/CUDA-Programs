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

    float* d_array, *d_out, *d_sum;

    cudaMalloc((void**)&d_array, Size*sizeof(float));
    cudaMalloc((void**)&d_out, ceil(Size*1.0/1024)*sizeof(float));
    cudaMalloc((void**)&d_sum, sizeof(float));

    cudaMemcpy(d_array, h_array, sizeof(float) * Size, cudaMemcpyHostToDevice);

    float h_sum;

    Array_Add <<<ceil(Size*1.0/1024), 1024, 1024*sizeof(float)>>> (d_out, d_array, Size);

    Array_Add <<<1, 1024, 1024*sizeof(float)>>> (d_sum, d_out, ceil(Size*1.0/1024));

    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_array);
    cudaFree(d_out);
    cudaFree(d_sum);

    return h_sum;
}

__global__ void Dot_Product(float* d_A, float* d_B, float* d_Prod, int Size)
{
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(id<Size)
        d_Prod[id] = d_A[id] * d_B[id];
}

int main()
{
    int Array_Size;
    cout << "Enetr the size of the two arrays.\n";
    cin >> Array_Size;
    int Array_Bytes = Array_Size * sizeof(float);
    float h_A[Array_Size], h_B[Array_Size], h_Prod[Array_Size];
    for(int i=0; i<Array_Size; i++)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)i;
    }
    float *d_A, *d_B, *d_Prod;

    cudaMalloc((void**)&d_A, Array_Bytes);
    cudaMalloc((void**)&d_B, Array_Bytes);
    cudaMalloc((void**)&d_Prod, Array_Bytes);

    cudaMemcpy(d_A, h_A, Array_Bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Array_Bytes, cudaMemcpyHostToDevice);

    Dot_Product<<<ceil(Array_Size*1.0/1024), 1024>>>(d_A, d_B, d_Prod, Array_Size);

    cudaMemcpy(h_Prod, d_Prod, Array_Bytes, cudaMemcpyDeviceToHost);

    /*for(int i=0; i<Array_Size; i++)
        cout << h_Prod[i] << " ";*/
    
    float Dot_Prod = Find_Sum_GPU(h_Prod, Array_Size);

    cout << "\nThe dot product is " << Dot_Prod << endl;

    cudaFree(h_A);
    cudaFree(h_B);
    cudaFree(h_Prod);
}