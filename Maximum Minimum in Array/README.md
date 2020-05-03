## Finding the maximum and minimum values in an array

<ul>
<li>This program ccomputes the maximum and minimum values in a large array using CUDA parallel programming. </li>
<li>The program populates the input array with random numbers in the range 0 to 10^9. </li>
<li>The maximum and minimum in the array are then computed on both the CPU and the GPU. </li>
<li>The GPU kernel uses the parallel reduction technique to compute the result. </li>
<li>The CPU and the GPU results are then compared to check the correctness of our result</li>
<li>In the GPU calculation, for upto 2^20 numbers, all the computation is done by only the GPU. For more than 2^20 numbers, the GPU computes local minimums and local maximums in blocks of 2^20 numbers, and the remaining calculation of global minimum and maximum is done by the CPU itself. </li>
</ul>

### How to Run

```
nvcc MaxMin.cu -o MaxMin
./MaxMin
```
