## Addition of two large integer matrices

This program adds corresponding elements of two large integer arrays parallely using CUDA parallel programming. It also compares the time taken by the GPU to perform the addition using parallel threads, with the time taken by the CPU to perform the same addition in a serial fashion.

### How to Run

```
nvcc Matrix_Add.cu  
./a.out 
```