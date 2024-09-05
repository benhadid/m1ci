#include <algorithm>
#include <string>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "CycleTimer.h"
#include "matrixMultiply_ispc.h"

#define VECT_SIZE 4

typedef float v4sf __attribute__((vector_size(VECT_SIZE * sizeof(float))));

/* Nous incluons ici les 6 variantes de l'ordre des boucles en tant que
   fonctions séparées, puis les appelons à l'aide de pointeurs de fonction.
   La raison d'avoir des fonctions séparées qui sont presque identiques est
   d'éviter de compter tout traitement superflu dans le temps de calcul.
   Cela inclut les accès E / S (printf) et les conditions (if / switch).
   Les accès d'E / S sont lents et les instructions conditionnelles / de branchement
   peuvent fausser les résultats.
*/

#define BS 32

void block_transpose(int M, int N, float *src, float *dst)
{
    for (int i = 0; i < M; i += BS)
        for (int j = 0; j < N; j += BS)
            for (int k = i; k < i + BS; ++k)
                for (int l = j; l < j + BS; ++l)
                    dst[k + l * M] = src[l + k * N];
}

void naive_multMat(int M, int N, int K, float *A, float *B, float *C)
{
    int i, j, k;

    for (i = 0; i < M; ++i)
        for (j = 0; j < N; ++j)
        {
            float val = 0;
            for (k = 0; k < K; ++k)
            {
                val += A[k + i * K] * B[j + k * N];
            }
            C[j + i * N] = val;
        }
}

void block_multMat(int M, int N, int K, float *A, float *B, float *C)
{
    for (int i = 0; i < M; i += BS)
        for (int j = 0; j < N; j += BS)
            for (int k = 0; k < K; k += BS)
                for (int ic = i; ic < i + BS; ++ic)
                    for (int jc = j; jc < j + BS; ++jc)
                    {
                        float val = 0;
                        for (int kc = k; kc < k + BS; ++kc)                        
                            val += A[kc + ic * K] * B[jc + kc * N];
                        
                        C[jc + ic * N] += val;
                    }
}

void transposed_multMat(int M, int N, int K, float *A, float *Bt, float *C)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
        {
            float val = 0;
            for (int k = 0; k < K; ++k)
                val += A[k + i * K] * Bt[k + j * K];
            C[j + i * N] = val;
        }
}

void transposed_vectorized_multMat(int M, int N, int K, float *A, float *Bt, float *C)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
        {
            v4sf cval = {0, 0, 0, 0};
            //   #pragma GCC unroll VECT_SIZE
            for (int k = 0; k < K; k += VECT_SIZE)
            {
                v4sf aval = *((v4sf *)&A[k + i * K]);
                v4sf bval = *((v4sf *)&Bt[k + j * K]);
                cval += aval * bval;
            }
            float val = 0;
#pragma GCC unroll 4
            for (int k = 0; k < VECT_SIZE; ++k)
                val += cval[k];
            C[j + i * N] = val;
        }
}

void transposed_vectorized_parallel_multMat(int M, int N, int K, float *A, float *Bt, float *C)
{
#pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
        {
            v4sf cval = {0, 0, 0, 0};
            // #pragma GCC unroll VECT_SIZE
            for (int k = 0; k < K; k += VECT_SIZE)
            {
                v4sf aval = *((v4sf *)&A[k + i * K]);
                v4sf bval = *((v4sf *)&Bt[k + j * K]);
                cval += aval * bval;
            }
            float val = 0;
#pragma GCC unroll 4
            for (int k = 0; k < VECT_SIZE; ++k)
                val += cval[k];
            C[j + i * N] = val;
        }
}

void block_transposed_vectorized_parallel_multMat(int M, int N, int K, float *A, float *Bt, float *C)
{

#pragma omp parallel for collapse(3)
    for (int i = 0; i < M; i+=BS)
        for (int j = 0; j < N; j+=BS)
            for(int k = 0; k < K; k += BS)               
                 ispc::mxm(A, Bt, C, M, N, K, i, j, k, BS);
}

bool differ(float *C, float *Cr, int size)
{
    float err = 0.0;
    int   idx = -1;

    for (int i = 0; i < size; ++i) {
        float val = C[i] - Cr[i];
        val *= val;

        if(val > err)
        {
            err = val;
            idx = i;
        }
    }

    err = sqrt(err);

    //bool isclose =  err <= 1.e-9 * std::max( std::abs(  C[idx]   ), std::abs(  Cr[idx] ));
    bool isclose =  err <= (1.e-8 + 1.e-5 * std::abs(Cr[idx]));

    return !isclose; // epsilon
}

/* utilise les fonctionnalités de chronométrage dans sys/time.h */
int main(int argc, char **argv)
{
    int N = 1024; // 4096;
    int M = 768;
    int K = 768; // 2048;

    void (*orderings[])(int, int, int, float *, float *, float *) =
        {&naive_multMat, &block_multMat, &transposed_multMat, &transposed_vectorized_multMat, &transposed_vectorized_parallel_multMat, &block_transposed_vectorized_parallel_multMat};
    std::string names[] = {"Naive", "Blocked (B)", "Transposed (T)", "T + Vectorized (V)", "T + V + P", "B + T + V + P"};

    float *A = (float *)aligned_alloc(sizeof(float), M * K * sizeof(float));
    float *B = (float *)aligned_alloc(sizeof(float), K * N * sizeof(float));
    float *Bt = (float *)aligned_alloc(sizeof(float), N * K * sizeof(float));
    float *C = (float *)aligned_alloc(sizeof(float), M * N * sizeof(float));
    float *Cr = (float *)aligned_alloc(sizeof(float), M * N * sizeof(float));

    /* remplir les matrices avec des nombres aléatoires */
    for (int i = 0; i < M * K; i++)
        A[i] = drand48() * 2 - 1;
    for (int i = 0; i < K * N; i++)
        B[i] = drand48() * 2 - 1;

    double clockedTime[5];

    clockedTime[0] = __DBL_MAX__;
    for (int j = 0; j < 3; ++j)
    {
        memset(C, 0, M * N * sizeof(float));
        double startTime = CycleTimer::currentSeconds();
        (*orderings[0])(M, N, K, A, B, C);
        double endTime = CycleTimer::currentSeconds();
        clockedTime[0] = std::min(clockedTime[0], endTime - startTime);
    }
    memcpy(Cr, C, M * N * sizeof(float)); // make a copy from serial compute for comparison
    printf("[%-34s]: [%.3f] ms\n", names[0].c_str(), clockedTime[0] * 1000);

    for (int i = 1; i < 2; i++)
    {
        clockedTime[i] = __DBL_MAX__;
        for (int j = 0; j < 3; ++j)
        {
            memset(C, 0, M * N * sizeof(float));
            double startTime = CycleTimer::currentSeconds();
            (*orderings[i])(M, N, K, A, B, C);
            double endTime = CycleTimer::currentSeconds();
            clockedTime[i] = std::min(clockedTime[i], endTime - startTime);
        }
        if (differ(C, Cr, M * N))
        {
            printf("Compute from \"%s\" differs from serial\n", names[i].c_str());
            exit(1);
        }
        printf("[%-34s]: [%.3f] ms\n", names[i].c_str(), clockedTime[i] * 1000);
    }

    for (int i = 2; i < 6; i++)
    {
        clockedTime[i] = __DBL_MAX__;
        for (int j = 0; j < 3; ++j)
        {
            memset(C, 0, M * N * sizeof(float));
            double startTime = CycleTimer::currentSeconds();
            block_transpose(K, N, B, Bt);
            (*orderings[i])(M, N, K, A, Bt, C);
            double endTime = CycleTimer::currentSeconds();
            clockedTime[i] = std::min(clockedTime[i], endTime - startTime);
        }
        if (differ(C, Cr, M * N))
        {
            printf("Compute from \"%s\" differs from serial\n", names[i].c_str());
            exit(1);
        }
        printf("[%-34s]: [%.3f] ms\n", names[i].c_str(), clockedTime[i] * 1000);
    }

    for (int i = 1; i < 6; ++i)
    {
        printf("\t\t(%.2fx speedup from \"%s\")\n", clockedTime[0] / clockedTime[i], names[i].c_str());
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
