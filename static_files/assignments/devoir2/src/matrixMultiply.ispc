
export inline void mxm(uniform float * uniform A,
 uniform float * uniform Bt,
 uniform float * uniform C,
 uniform int M,
 uniform int N,
 uniform int K,
 uniform int i,
 uniform int j,
 uniform int k,
 uniform int BS
)
{    

        for (uniform int ic = i; ic < i + BS; ic++)
            for (uniform int jc = j; jc < j + BS; jc++)
            {
                float val = 0.0;          
                
                foreach(kc = k ... k + BS)                
                    val += A[ic * K + kc] * Bt[jc * K + kc];
                
                C[ic * N + jc] += reduce_add(val);
            }
  
}


