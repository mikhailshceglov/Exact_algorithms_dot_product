#include "pichat.hpp"
#include <vector>

using namespace std;

void pichat_sum(vector<double>& vec){
    for (int i = vec.size() - 1; i > 0; i--){
        double s = vec[i-1] + vec[i];          // Округлённая сумма
        double r = (vec[i-1] - s) + vec[i];   // Точный остаток
        vec[i-1] = s;
        vec[i] = r;
    }
}

double pichat_dot_product(const vector<double>& a, const vector<double>& b){
    vector<double> products(a.size());
    for (int i = 0; i < a.size(); i++){
        products[i] = a[i] * b[i];
    }
    
    for (int k = 0; k < products.size() - 1; k++){
        pichat_sum(products);
    }
    
    return products[0];
}