#include<iostream>
#include<time.h>
using namespace std;

__global__ void Matrix_Add(int* d_A, int* d_B, int* d_Sum)
{
    int i = blockIdx.y;
    int j = threadIdx.x;
    int id = (i * blockDim.x) + j;
    *(d_Sum + id) = *(d_A + id) + *(d_B + id);
}

int main()
{
    const int Rows = 4;
    const int Cols = 4;
    const int Size = Rows * Cols * sizeof(int);

    int h_A[Rows][Cols], h_B[Rows][Cols], h_Sum[Rows][Cols];
    for(int i=0; i<Rows; i++)
    {
        for(int j=0; j<Cols; j++)
        {
            h_A[i][j] = (i * Cols) + j + 1;
            h_B[i][j] = (i * Cols) + j + 1;
        }
    }

    int *d_A, *d_B, *d_Sum;

    clock_t start, end;

    cudaMalloc((void**)&d_A, Size);
    cudaMalloc((void**)&d_B, Size);
    cudaMalloc((void**)&d_Sum, Size);

    cudaMemcpy(d_A, h_A, Size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Size, cudaMemcpyHostToDevice);

    start = clock();

    Matrix_Add<<<dim3(1, Rows, 1), dim3(Cols, 1, 1)>>>(d_A, d_B, d_Sum);

    end = clock();
    
    cudaMemcpy(h_Sum, d_Sum, Size, cudaMemcpyDeviceToHost);

    cout << "\nGPU time required for addition of two 500 by 500 matrices : " << (double)(end - start) << endl;
    
    
    for(int i=0; i<Rows; i++)
    {
        for(int j=0; j<Cols; j++)
            cout << h_Sum[i][j] << "\t";
        cout << endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_Sum);

    int h_Sum_CPU[Rows][Cols];

    start = clock();

    for(int i=0; i<Rows; i++)
    {
        for(int j=0; j<Cols; j++)
            h_Sum_CPU[i][j] = h_A[i][j] + h_B[i][j];
    }

    end = clock();

    cout << "\nCPU time required for addition of two 500 by 500 matrices : " << (double)(end - start) << endl;
}