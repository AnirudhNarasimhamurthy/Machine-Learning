/* 
 * Author: Mengyang Wang
 *
 */


#include <iostream>
#include <cstring>
#include <ctime>
#include <cstdlib>

using namespace std;

template<int d>
float wtx(float (&w)[d], float (&x)[d])
{
    float res = 0.0;
    for(int i=0; i<d; i++)
        res += w[i] * x[i];

    return res;
}

template<int d>
void initw(float (&w)[d])
{
    srand (time(NULL));
    float lo = -1.0;
    float hi = 1.0;   
    for(int i=0; i<d; i++)
    {
        w[i] = lo + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(hi-lo)));
    }
}

//init bias b
void initb(float & b)
{
    srand (time(NULL));
    float lo = -1.0;
    float hi = 1.0;    
    b = lo + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(hi-lo)));
}


int main(int argc, char ** argv)
{
    float x [][4] = {{1, 0, 0, 0},
                        {1, 1, 0, 0},
                        {1, 0, 1, 1},
                        {0, 1, 0, 0},
                        {0, 1, 1, 0},
                        {1, 1, 1, 0},
                        {0, 1, 1, 1}};

    const int y_original [] = {0, 0, 1, 0, 0, 0, 1};

    //dimension of weight vector
    const int d = 4;

    //number of data points 
    int n = sizeof(y_original) / sizeof(int);

    int y[n];
    for(int i=0; i<n; i++)
        y[i] = y_original[i] * 2 - 1;

 
    

    float r[]= {0.1, 0.3, 0.5, 0.75, 1.0};

    //train and count # mistake
    for(int k = 0; k < sizeof(r) / sizeof(float); k++)
    {
        float w[d]; 
        initw(w);
        float b;
        initb(b);

        int cnt = 0;

        for(int i = 0; i < n; i++)
        {

            if( (wtx(w, x[i]) + b) * y[i] <= 0.0 )
            {
                for(int j=0; j<d; j++)
                    w[j] += (r[k] * y[i] * x[i][j]);
                
                b += (r[k] * y[i]);
                cnt++;
            }
        }

        cout << "r = " << r[k] << ", w = [" << w[0] << ", " << w[1] << ", " << w[2] << ", " << w[3] << "], b = " 
             <<  b << ", # mistake = "<< cnt <<endl;
    }

    
    return 0;
}
