#include "fma.hpp"
#include <cmath>
#include <cstdint>
#include <vector>
#include <cstdio>

using namespace std;

double fma_dot_product(const vector<double>& a, const vector<double>& b){
    vector<double> table;
    double fma_sum = 0.0;
    
    for (int i = 0; i < a.size(); i++){
        fma_sum = fma(a[i], b[i], fma_sum);
    }

    return fma_sum; 
}