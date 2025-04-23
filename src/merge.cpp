#include "merge.hpp"
#include <cmath>
#include <cstdint>
#include <vector>
#include <cstdio>

using namespace std;

struct DoubleDouble{
    double hi; // Основная часть
    double lo; // Корректирующая часть
};

void dod_add(DoubleDouble& sum, double x, double err){
    double s1 = sum.hi + x;
    double e1 = (sum.hi - s1) + x;
    
    double s2 = s1 + (e1 + err + sum.lo);
    double e2 = (s1 - s2) + (e1 + err + sum.lo);
    
    sum.hi = s2;
    sum.lo = e2;
}

double dod_get(const DoubleDouble& sum){
    double result = sum.hi + sum.lo;
    double residual = (sum.hi - result) + sum.lo;
    return result + residual;
}

// Точное умножение с использованием FMA
pair<double, double> exact_multiply(double a, double b){
    double prod = a * b;
    double err = fma(a, b, -prod); // Вычисление ошибки
    return {prod, err};
}

double merge_dot_product(const vector<double>& a, const vector<double>& b){
    DoubleDouble sum{0.0, 0.0};
    for (int i = 0 ; i < a.size(); i++){
        pair<double,double> sump = exact_multiply(a[i], b[i]);
        dod_add(sum, sump.first, sump.second);
    }
    return dod_get(sum);
}
