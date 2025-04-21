#include "long_accumulator.hpp"
#include <vector>

using namespace std;

struct DoubleDouble{
    double hi; // Основная часть
    double lo; // Корректирующая часть
};

void dd_add(DoubleDouble& sum, double x){
    double s = sum.hi + x;
    double e = (sum.hi - s) + x; // Компенсация ошибки
    sum.hi = s;
    sum.lo += e;
}

double dd_get(const DoubleDouble& sum){
    return sum.hi + sum.lo;
}

double long_accumulator_dot_product(const vector<double>& a, const vector<double>& b){
    DoubleDouble sum{0.0, 0.0};
    for (int i = 0; i < a.size(); i++){
        dd_add(sum, a[i] * b[i]);
    }
    return dd_get(sum);
}