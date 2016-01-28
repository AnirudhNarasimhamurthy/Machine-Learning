/* 
 * process.h: 
 *     header file that include all the necessary fuctions for perceptron and its variants
 * 
 * Author: Mengyang Wang
 */


#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <ctime>

using namespace std;


//a utility function for shuffle
void swap (int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

//shuffle function
//template<int n>
void shuffle_arr ( int arr[], int n )
{

    srand ( time(NULL) );
    for (int i = n-1; i > 0; i--)
    {
        int j = rand() % (i+1);
        swap(&arr[i], &arr[j]);
    }
}

//init weight vector w
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


//convert vector<int>
//example: {1, 3, 5} => {1, 0, 1, 0, 1, ...}
//not input vector feature idx starting from 1, not 0
template<int d>
void vec2arr(vector<int> vec, float (&arr)[d])
{
    memset(arr, 0, sizeof(float) * d);
    for(int i=0; i<vec.size(); i++)
    {
        int idx = vec[i];
        arr[idx-1] = 1.0;
    }
}


//calculate w tranpose dot product with x
template<int d> 
float wtx(float (&w)[d], float (&x)[d])
{
    float res = 0.0;
    for(int i=0; i<d; i++)
        res += w[i] * x[i];

    return res;
}

//load data, convert x to vector<vector<int> >; convert y to vector<int>
void load(vector<vector<int> > & x, vector<int> & y, string filename)
{
    ifstream ifs(filename.c_str());
    if(ifs.eof())
    {
        cerr << "Unable to load file: " << filename << ", make sure it is on the same directory with the code." << endl; 
        return;
    }
    while(!ifs.eof())
    {
        string line;
        getline(ifs, line);

        size_t pos = 0;
        string token;
        string delimiter = " ";

        //process each line, extract vector x and label y
        vector<int> vec;

        while (  (pos = line.find(delimiter)) != string::npos  ) 
        {
            token = line.substr(0, pos);
            //cout << token << endl;
            if(token.size() == 2) //label
            {
                if(token[0] == '+')
                    y.push_back(1);
                else 
                    y.push_back(-1);
            }
            else
            {
                size_t p = token.find(":");
                string sidx = token.substr(0, p);
                vec.push_back( atoi(sidx.c_str()) );
            }

            line.erase(0, pos + delimiter.length());
        }
        if(vec.size() > 0)
            x.push_back(vec);
    }
}



template<int d>
void train(vector<vector<int> > & trainx, 
           vector<int>  & trainy, 
           float (&w)[d], float &b, 
           float mu, float r, 
           int epoch, bool shuffle, bool use_eta)
{

    int cnt = 0;
    int sz = trainx.size();

    int order[sz];
    for(int i=0; i<sz; i++)
        order[i] = i;
    
    float eta = r;

    for(int i=0; i<epoch; i++)
    {
        if(shuffle)
            shuffle_arr ( order, sz );
        
        for(int j = 0; j < sz; j++)
        {
            int idx = order[j];
            vector<int> xvec = trainx[idx];
            float x[d];
            memset(x, 0.0, sizeof(float) * d);
            vec2arr(xvec, x);
            
            float ret = wtx(w, x);
            float y = (float)trainy[idx];

            if( y * (ret + b) <= mu )
            {
                if(use_eta)
                    eta = (mu - y * (ret + b)) / (wtx(x, x) + 1.0);
                
                //update w
                for(int k = 0 ; k < d; k++)
                    w[k] += (eta * y * x[k]);

                //update b
                b += (eta * y);

                cnt++;
            }
        }
    }
    cout << "# update: " << cnt << ", accuracy on training set: " << ( (float)sz * epoch - cnt) / (epoch * sz);
}

template<int d>
void test(vector<vector<int> > testx, vector<int> testy, float (&w)[d], float &b, float mu)
{
    int cnt = 0;
    for(int k = 0; k < testx.size(); k++)
    {
        vector<int> xvec = testx[k];
        float x[d];
        memset(x, 0.0, sizeof(float) * d);
        vec2arr(xvec, x);   
        float ret = wtx(w, x);
        float y = (float)testy[k];
        if( y * (ret + b) <= mu )
            cnt++;
    }
    cout << ", accuracy on test set: " << ((float)testx.size() - cnt) / testx.size() << endl << endl;
}