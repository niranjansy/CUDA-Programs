#include<iostream>
using namespace std;

__global__ void Array_Add(float* d_A, float* d_B, float* d_Sum)
{
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    d_Sum[id] = d_A[id] + d_B[id];
}

int main()
{
    const int Array_Size = 320000;
    const int Array_Bytes = Array_Size * sizeof(float);
    float h_A[Array_Size], h_B[Array_Size], h_Sum[Array_Size];
    for(int i=0; i<Array_Size; i++)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)i;
    }
    float *d_A, *d_B, *d_Sum;

    //Measuring performance
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void**)&d_A, Array_Bytes);
    cudaMalloc((void**)&d_B, Array_Bytes);
    cudaMalloc((void**)&d_Sum, Array_Bytes);

    cudaMemcpy(d_A, h_A, Array_Bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Array_Bytes, cudaMemcpyHostToDevice);

    //Start of performance measurement
    cudaEventRecord(start);

    Array_Add<<<625, 512>>>(d_A, d_B, d_Sum);

    //End of performance measurement
    cudaEventRecord(stop);

    //Block CPU execution until the event "stop" is recorded
    cudaEventSynchronize(stop);

    //Print the time taken in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "The total time taken is " <<  milliseconds << " milliseconds.\n";

    cudaMemcpy(h_Sum, d_Sum, Array_Bytes, cudaMemcpyDeviceToHost);

    for(int i=0; i<Array_Size; i++)
        cout << h_Sum[i] << " ";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_Sum);
}