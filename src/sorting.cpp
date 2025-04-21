#include "sorting.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

// Компаратор для сортировки по убыванию экспоненты
bool compare_by_exponent(double a, double b){
    int exp_a, exp_b;
    frexp(a, &exp_a);
    frexp(b, &exp_b);
    return exp_a > exp_b;
}

double sorting_dot_product(const vector<double>& a, const vector<double>& b){
    vector<double> products(a.size());
    for (int i = 0; i < a.size(); i++){
        products[i] = a[i] * b[i];
    }
    
    std::sort(products.begin(), products.end(), compare_by_exponent);
    
    double sum = 0.0;
    for (auto val : products) sum += val;
    
    return sum;
}