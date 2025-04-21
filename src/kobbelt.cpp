#include "kobbelt.hpp"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>

using namespace std;

struct NumberInfo {
    double value;
    int exp;
    bool is_even;
};

NumberInfo get_number_info(double x){
    NumberInfo info;
    info.value = x;
    int exp;
    frexp(x, &exp);
    info.exp = exp;
    uint64_t bits;
    memcpy(&bits, &x, sizeof(double));
    info.is_even = (bits & 1) == 0;
    return info;
}

void insert_into_table(vector<double>& table, double x){
    NumberInfo info = get_number_info(x);
    size_t index = 2 * (info.exp + 1024) + (info.is_even ? 0 : 1);
    if (index >= table.size()) table.resize(index + 1, 0.0);
    table[index] += x;
}

double kobbelt_dot_product(const vector<double>& a, const vector<double>& b){
    std::vector<double> table;
    for (int i = 0; i < a.size(); i++){
        double product = a[i] * b[i];
        insert_into_table(table, product);
    }
    double sum = 0.0;
    for (auto val : table) sum += val;
    return sum;
}