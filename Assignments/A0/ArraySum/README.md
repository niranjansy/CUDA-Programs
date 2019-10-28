## Addition of all numbers in an array

This program adds up all elements of an array in CUDA parallel programming, using the parallel reduce algorithm. This algorithm works in logarithmic step complexity. 
(The program works only for arrays of size atmost 1048576 (1024*1024)). 

### How to Run

```
nvcc Array_Sum.cu -o Array_Sum
./Array_Sum 
```