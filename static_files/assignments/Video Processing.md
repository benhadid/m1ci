## Application: Video Processing

For this project, we will be applying convolutions to a real world application: video processing. Convolutions can blur, sharpen, or apply other effects to videos. This is possible because individual frames in a video can be treated as matrices of red, blue, and green values constructing the color for a given pixel. For simplicity, we're only working with grayscale videos, so that there's only one value per pixel (as opposed to one value for red, one for green, one for blue). As such, we can perform any matrix operation on each video frame, one of them being convolution.

When we convolve a matrix with an image, the matrix we use will have a major impact on the outcome ranging from sharpening to blurring an image. The matrices that we provide in this project will blur or sharpen your video frames. For each pixel, we compute a weighted average using the pixel itself and the pixels near it. By averaging many pixels together, this will smoothen any difference between their values, resulting in a blur. This is referred to as “Gaussian Blur” and is exactly how your phone blurs photos.


## Vectors

In this project, a vector is represented as an array of `int32_t`s, or a `int32_t *`.

## Matrices
In this project, we provide a type `matrix_t` defined as follows:

```C
typedef struct {
  uint32_t rows;
  uint32_t cols;
  int32_t *data;
} matrix_t;
```

In `matrix_t`, `rows` represents the number of rows in the matrix, `cols` represents the number of columns in the matrix, and `data` is a 1D array representation of the matrix stored in **row-major** format (similar to lab 1). For example, the matrix `[[1, 2, 3], [4, 5, 6]]` (as in Python) would be stored as `[1, 2, 3, 4, 5, 6`.

## `.bin` files

Matrices are stored in `.bin` files as a consecutive sequence of 4-byte integers. The first and second integers in the file indicate the number of rows and columns in the matrix, respectively. The rest of the integers store the elements in the matrix in row-major order.

To view matrix files, you can run `xxd -e matrix_file.bin`, replacing `matrix_file.bin` with the matrix file you want to examine. The output should look something like this:

```shell
00000000: 00000003 00000003 00000001 00000002  ................
00000010: 00000003 00000004 00000005 00000006  ................
00000020: 00000007 00000008 00000009           ............
```

The left-most column indexes the bytes in the file (e.g. the third row starts at the `0x20`th byte of the file). The dots on the right display the bytes in the file as ASCII, but since these bytes don't correspond to printable ASCII characters so only dot placeholders appear.

The actual contents of the file are listed in 4-byte blocks, 4 per row. The first row has the numbers 3 (row count), 3 (column count), 1 (first element), and 2 (second element). This is a 3x3 matrix with elements [1, 2, 3, 4, 5, 6, 7, 8, 9].

## Task 1: Naive Convolutions
In this project, you will implement and optimize 2D convolutions, which is a mathematical operation that has a wide range of applications. Don't worry if you've never seen convolutions before, it can be simplified to a series of dot products (more on this in task 1.3).

Convolution is a special way of multiplying two vectors or two matrices together. This leads to many different applications that you'll explore in this project, but first, here are the mechanics for how convolution is done:

A convolution is when you want to convolve two vectors or matrices together, matrix A and matrix B. We will assume that matrix B is always smaller than matrix A.

1. You begin by flipping matrix B in both dimensions. Note that flipping matrix B in both dimensions is NOT the same as transposing the matrix. Flipping an MxN matrix results in an MxN matrix. Transpose results in an NxM matrix.

[img1] next to [imag2]

2. Once flipped horizontally and vertically, overlap matrix B in the top left corner of matrix A. Perform an element-wise multiplication of where the matrices overlap and then add all of the results together to get a single value. This is the top left entry in your resultant matrix.

3. Slide matrix B to the right by 1 and repeat this process. This continues until any part of matrix B no longer overlaps with matrix A. When this happens, move matrix B to first column of matrix A and down by 1 row.

4. Repeat the entire process until reaching the bottom right corner of matrix A. You have now convolved matrix A and B. (click the image for a larger version)

You can assume that the height and width of matrix B are less than or equal to the height and width of matrix A.

Note: The output matrix has different dimensions from its input matrices. We'd recommend working out some examples to see how the dimensions of the output matrix is related to the input matrices.

Implement convolve in `compute_naive.c`. You may assume that `b_matrix` is smaller or equal to `a_matrix` (that is, if `a_matrix` is `m` by `n` and `b_matrix` is `k` by `l`, then `k < m` and `l < n`).

convolve in compute_naive.c
input arguments:
matrix_t* a_matrix  pointer to matrix A
matrix_t* b_matrix  pointer to matrix B
output:
matrix_t** output_matrix  return a pointer to the result matrix - you must allocate memory for output_matrix
return:
int        0 if successful, -1 if there are errors

---
testing
---


Task 2: Optimization thru  SIMD

Optimize your naive solution using SIMD instructions in compute_optimized.c. Not all functions can be optimized using SIMD instructions. For this project, use vectors that can store 8 integers and perform 8 operations at once. Don't forget to implement any tail case(s)!

---
testing
---

Task 3: Optimization thru  OpenMP

Optimize your solution from task 2 using OpenMP directives in compute_optimized.c. Not all functions can be optimized using OpenMP directives. You can find more information on OpenMP directives on the [OpenMP summary card]().

---
testing
---

Task 4: Algorithmic Optimizations
If your solution doesn't meet your desired speedups, there is probably room for algorithmic optimizations!  The specific algorithmic optimizations depend on your existing solution and algorithm ! (Hint: think of caching! and lab1)

Reduce function calls: Function calls are slow because the program must set up a stack frame and jump to a different part of code. If some part of your code is repeatedly calling a function, you could try to reduce the number of times the function is called.

Loop unrolling: You can manually reduce the number of loop iterations your program needs to execute. See [this link](https://en.wikipedia.org/wiki/Loop_unrolling#Simple_manual_example_in_C) for a simple example of loop unrolling.

Cache blocking: You can rearrange how data is stored in memory to improve locality for better cache performance. For example, accessing a column of data in a row-major matrix requires skipping through elements in the matrix, but if the same matrix were stored in column-major order, the column of data would be located in one continuous block of memory.

---
testing
---

Task 5: MPI 










