#include<iostream>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include <cmath>
#include <cuda.h>
using namespace std;
#define ll long long int

__global__ void Inclusive_Scan(ll *d_in, ll* d_out, ll Size, ll i)
{
    ll id = blockIdx.x * blockDim.x + threadIdx.x;
    ll step = 1 << i;
    if(id < Size)
    {
        if(id >= step)
        {
            d_out[id] = d_in[id] + d_in[id-step];
        }
        else
        {
            d_out[id] = d_in[id];
        }
    }
    __syncthreads();
}

int main()
{
    ll *h_in, *h_out;

    ll Size;
    cout << "Enter size of the array.\n";
    cin >> Size;
    ll Array_Bytes = Size * sizeof(ll);
    h_in = (ll*)malloc(Array_Bytes);
    h_out = (ll*)malloc(Array_Bytes);

    srand(time(0));
    for(ll i=0; i<Size; i++)
    {
        h_in[i] = rand()%10;
    }

    /*cout << "Input Array : \n";
    for(ll i=0; i<Size; i++)
        cout << h_in[i] << " ";
    cout << endl;*/

    ll *d_in, *d_out;

    cudaMalloc((void**)&d_in, Array_Bytes);
    cudaMalloc((void**)&d_out, Array_Bytes);

    cudaMemcpy(d_in, h_in, Array_Bytes, cudaMemcpyHostToDevice);

    ll iterations = (ll)floor(log2((double)Size)) + 1;

    for(ll i=0; i<iterations; i++)
    {
        Inclusive_Scan <<< (int)ceil(1.0*Size/1024), 1024>>> (d_in, d_out, Size, i);
        cudaMemcpy(d_in, d_out, Array_Bytes, cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(h_out, d_out, Array_Bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    /*cout << "Inclusive Scan Array : \n";
    for(ll i=0; i<Size; i++)
        cout << h_out[i] << " ";
    cout << endl;*/

    ll *pref;
    pref = (ll*)malloc(Array_Bytes);
    pref[0] = h_in[0];
    for(ll i=1; i<Size; i++)
        pref[i] = pref[i-1] + h_in[i];

    /*for(ll i=0; i<Size; i++)
        cout << pref[i] << " ";
    cout << endl;*/

    ll flag = 0;
    for(ll i=0; i<Size; i++)
    {
        if(h_out[i] != pref[i])
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