## Finding dot product of two large vectors

This program calculates the dot product of two vectors using CUDA parallel programming. The program first computes the products of every pair of corresponding elements of the two arrays using the map operation and then uses the parallel reduce algorithm to sum up all elements of the resulting array.

### How to Run

```
nvcc DotProduct.cu -o DotProduct
./DotProduct 
```