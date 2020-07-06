#include<iostream>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include <cmath>
#include <cuda.h>
using namespace std;
#define ll long long int

const int Block_Size = 1024;

// This GPU kernel does blockwise in-place scan 
__global__ void Inclusive_Scan(ll *d_in, ll* d_out)
{
    __shared__ ll sh_array[Block_Size];

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Copying data from global to shared memory
    sh_array[tid] = d_in[id];

    __syncthreads();

    // Performing block-wise in-place Hillis-Steele scan
    // In the ith step (starting from 0), to every element we add the element that is (2^i) places to its left, or retain the same element if it does not have so many places to its left
    for(int step = 1; step <= Block_Size; step *= 2)
    {
        if(tid >= step)
        {
            ll temp = sh_array[tid-step];
            __syncthreads();
            sh_array[tid] += temp;
        }
        __syncthreads();
    }
    __syncthreads();

    // Copying the scan result back into global memory
    d_in[id] = sh_array[tid];
    // d_in now contains blockwise scan result
    __syncthreads();

    // Storing the blockwise sums into d_out, i.e, we store the last element in each block of scanned array in its corresponding position in d_out
    if(tid == (Block_Size - 1))
        d_out[bid] = d_in[id];

    __syncthreads();
}

// This GPU kernel adds the value d_out[id] to all values in the (id+1)th block of d_in
__global__ void Add(ll* d_in, ll* d_out)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;

    if(bid > 0)
        d_in[id] += d_out[bid-1];

    __syncthreads();
}

int main()
{
    ll *h_in, *h_scan;

    int Size;
    cout << "Enter size of the array.\n";
    cin >> Size;
    
    int Reduced_Size = (int)ceil(1.0*Size/Block_Size);  // The number of blocks that we need to launch
    int Array_Bytes = Size * sizeof(ll);
    int Reduced_Array_Bytes = Reduced_Size * sizeof(ll);

    h_in = (ll*)malloc(Array_Bytes);
    h_scan = (ll*)malloc(Array_Bytes);

    // Populating array with random numbers
    srand(time(0));
    for(ll i=0; i<Size; i++)
    {
        h_in[i] = rand()%10;
    }

    /*cout << "Input Array : \n";
    for(ll i=0; i<Size; i++)
        cout << h_in[i] << " ";
    cout << endl;*/

    ll *d_in, *d_out, *d_sum;

    // GPU Memory allocations
    cudaMalloc((void**)&d_in, Reduced_Size*Block_Size*sizeof(ll));  
    // Padding the input array to the next multiple of Block_Size. 
    // The scan algorithm is not dependent on elements past the end of the array, so we don't have to use a special case for the last block.
    cudaMalloc((void**)&d_out, Reduced_Array_Bytes);
    cudaMalloc((void**)&d_sum, sizeof(ll));

    // Copying input array from CPU to GPU
    cudaMemcpy(d_in, h_in, Array_Bytes, cudaMemcpyHostToDevice);

    Inclusive_Scan <<< Reduced_Size, Block_Size >>> (d_in, d_out);
    // After first kernel call, d_in has the blockwise scan results and d_out is an auxiliary array that has the blockwise sums

    // Second kernel call is done to scan the blockwise sums array 
    // Then the ith value in the resultant scanned blockwise sums array is added to every value in the (i+1)th block
    // This addition step is done in the Add kernel
    // This is required only if size of the array is greater than the block size
    if(Size > Block_Size)
    {
        Inclusive_Scan <<< 1, Block_Size>>> (d_out, d_sum);
        Add <<< Reduced_Size, Block_Size >>> (d_in, d_out);
    }

    // Copying the result back to the CPU
    cudaMemcpy(h_scan, d_in, Array_Bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    /*cout << "Inclusive Scan Array : \n";
    for(ll i=0; i<Size; i++)
        cout << h_scan[i] << " ";
    cout << endl;*/

    // CPU computation of the scan
    ll *pref;
    pref = (ll*)malloc(Array_Bytes);
    pref[0] = h_in[0];
    for(ll i=1; i<Size; i++)
        pref[i] = pref[i-1] + h_in[i];

    // Checking correctness of the result by comparing CPU and GPU results
    ll flag = 0;
    for(ll i=0; i<Size; i++)
    {
        if(h_scan[i] != pref[i])
        {
            flag = 1;
            break;
        }
    }
    if(flag == 0)
        cout << "Result computed correctly!\n";
    else
        cout << "Result wrong!\n";
}