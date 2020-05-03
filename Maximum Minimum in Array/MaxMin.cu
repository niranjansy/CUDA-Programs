#include<iostream>
#include<stdlib.h>
#include<math.h>
#include<time.h>
using namespace std;

// Implementation of the parallel reduce algorithm to find the maximum and minimum of all numbers in an array

__global__ void Array_MaxMin(int* d_max, int* d_min, int* d_array, int Size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    __shared__ int sh_array_max[1024];
    __shared__ int sh_array_min[1024];
    
    if(id < Size)
    {
        sh_array_max[tid] = d_array[id];
        sh_array_min[tid] = d_array[id];
    }
        
    __syncthreads();

    for(int s = 512; s>0; s = s/2)
    {
        __syncthreads();
        if(id>=Size || id+s>=Size)
            continue;
        if(tid<s)
        {
            sh_array_max[tid] = max(sh_array_max[tid], sh_array_max[tid+s]);
            sh_array_min[tid] = min(sh_array_min[tid], sh_array_min[tid+s]);
        }
    }

    __syncthreads();
    if(tid==0)
    {
        d_max[bid] = sh_array_max[tid];
        d_min[bid] = sh_array_min[tid];
    }
}

pair<int, int> Find_MaxMin_GPU(int h_array[], int Size)
{
    int* d_array, *d_out1, *d_out2, *d_max, *d_min, *d_dummy;

    int sub_size = (int)ceil(Size*1.0/1024);
    // The resultant number of blocks obtained on dividing the array into blocks of 1024 elements

    cudaMalloc((void**)&d_array, Size*sizeof(int));
    cudaMalloc((void**)&d_out1, sub_size*sizeof(int));
    cudaMalloc((void**)&d_out2, sub_size*sizeof(int));

    cudaMemcpy(d_array, h_array, sizeof(int) * Size, cudaMemcpyHostToDevice);

    Array_MaxMin <<<sub_size, 1024>>> (d_out1, d_out2, d_array, Size);
    // d_out1 now contains the local maximums in blocks of 1024 numbers
    // d_out2 contains local minimums in blocks of 1024 numbers

    int final_size = (int)ceil(sub_size*1.0/1024);
    // The resultant number of blocks obtained on dividing the array into blocks of 2^20 elements

    cudaMalloc((void**)&d_max, final_size*sizeof(int));
    cudaMalloc((void**)&d_min, final_size*sizeof(int));
    cudaMalloc((void**)&d_dummy, final_size*sizeof(int));

    Array_MaxMin <<<final_size, 1024>>> (d_max, d_dummy, d_out1, ceil(Size*1.0/1024));
    Array_MaxMin <<<final_size, 1024>>> (d_dummy, d_min, d_out2, ceil(Size*1.0/1024));
    // d_max and d_min contain local maximum and local minimum respectively in blocks of 2^20 numbers
    // The remaining maximums and minimums are computed on CPU itself

    int *h_max, *h_min;
    h_max = (int*)malloc(final_size*sizeof(int));
    h_min = (int*)malloc(final_size*sizeof(int));

    cudaMemcpy(h_max, d_max, final_size*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_min, d_min, final_size*sizeof(int), cudaMemcpyDeviceToHost);

    int M = h_max[0], m = h_min[0];
    for(int i=1; i<final_size; i++)
    {
        M = max(M, h_max[i]);
        m = min(m, h_min[i]);
    }

    cudaFree(d_array);
    cudaFree(d_out1);
    cudaFree(d_out2);
    cudaFree(d_min);
    cudaFree(d_max);
    cudaFree(d_dummy);

    return make_pair(M, m);
}

pair<int, int> Find_MaxMin_CPU(int* h_array, int Size)
{
    int Min, Max;
    Max = h_array[0];
    Min = h_array[0];
    for(int i=1; i<Size; i++)
    {
        Max = max(Max, h_array[i]); 
        Min = min(Min, h_array[i]);
    }
        
    return make_pair(Max, Min);
}

void RandomGenerator(int* h_array, int Size)
{
    srand(time(0));

    for(int i=0; i<Size; i++)
    {
        // rand() generates a random number between 0 and 32767
        h_array[i] = rand() * rand();
    }
        
}

int main()
{

    int Size;
    cout << "\nEnter the size of the array : ";
    cin >> Size;
    int *h_array;
    h_array = (int*)malloc(Size * sizeof(int));
    
    RandomGenerator(h_array, Size);

    pair<int, int> h_P = Find_MaxMin_GPU(h_array, Size);

    pair<int, int> P = Find_MaxMin_CPU(h_array, Size);

    /*cout << "The array elements are : \n";
    for(int i=0; i<Size; i++)
        cout << h_array[i] << " ";
    cout << endl;*/

    cout << "\nThe maximum and the minimum are " << h_P.first << " and " << h_P.second << endl;

    if(h_P == P)
        cout << "Result computed correctly.";
    else
        cout << "Result wrong!";
}