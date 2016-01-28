/* 
 * Author: Mengyang Wang
 *
 */

#include "process.h"

const int d = 123;

int main(int argc, char ** argv)
{
    bool shuffle = true;

    vector<vector<int> > trainx, testx;
    vector<int> trainy, testy;

    load(trainx, trainy, "a1a.train");
    load(testx, testy, "a1a.test");

    //experiment parameters
    //float mu[]= {0.0, 0.5, 1.0, 3.0, 5.0};
    float mu[]= {0.0, 0.5};
    //float r[]= {0.1, 0.2, 0.5, 0.75, 1.0};
    float r[]= {0.5};
    
    for(int i=0; i < sizeof(mu) / sizeof(float); i++)
    {
        for(int j=0; j < sizeof(r) / sizeof(float); j++)
        {
            if(i==0)
                cout << "Perceptron (mu = 0), r = " << r[j] << endl;
            else
                cout << "Margin Perceptron, mu = " << mu[i] << ", r = "<< r[j] << endl;

            //initialize w, b using random values between -1 to 1
            float w[d];
            initw(w);

            float b;
            initb(b);
    
            //training
            //void train(vector<vector<int> > & trainx, vector<int>  & trainy, float (&w)[d], float &b, float mu, float r, int epoch, bool shuffle, bool use_eta)
            train(trainx, trainy, w, b, mu[i], r[j], 3, shuffle, true);
            
            //testing
            //void test(vector<vector<int>> testx, vector<int> testy, float (&w)[d], float &b, float )
            test(testx, testy, w, b, mu[i]);

        }
    }


    return 0;
}