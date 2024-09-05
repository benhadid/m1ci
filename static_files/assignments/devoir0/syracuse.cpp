#include <iostream>
#include <omp.h>
#include "CycleTimer.h"

using namespace std;

#define TOTAL 2000000

int main(int, char *[])
{
    int pass=0, cur;
    //double start = omp_get_wtime();
    double start = CycleTimer::currentSeconds();
///////////////////// MODIFIER A PARTIR D'ICI UNIQUEMENT /////////////////////
    #pragma omp parallel for private(cur) reduction(+:pass)
    for(int i=0; i<TOTAL; i++) {
        cur=i;
        while (cur>1)
            cur=cur%2?3*cur+1:cur/2;
        pass++;
    }
///////////////////// MODIFIER JUSQU'ICI UNIQUEMENT /////////////////////
    //double end = omp_get_wtime();
    double end   = CycleTimer::currentSeconds();
    cout << pass << " out of " << TOTAL << "! (delta=" << TOTAL-pass << ")" << endl;
    cout << "ellapsed time: " << (end-start)*1000 << "ms" << endl;
}
