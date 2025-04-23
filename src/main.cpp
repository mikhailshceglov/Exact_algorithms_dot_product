#include "fma.hpp"
#include "kobbelt.hpp"
#include "long_accumulator.hpp"
#include "pichat.hpp"
#include "sorting.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>

using namespace std;

int main(){
    vector<double> a = {1.0, 2.0, 3.0, 1e300, 1e-300};
    vector<double> b = {4.0, 5.0, 6.0, 1e-300, 1e300};

    double res_fma = fma_dot_product(a, b);
    double res_kobbelt = kobbelt_dot_product(a, b);
    double res_long = long_accumulator_dot_product(a, b);
    double res_pichat = pichat_dot_product(a, b);
    double res_sorting = sorting_dot_product(a, b);

    cout << "FMA:          " << res_fma << "\n";
    cout << "Kobbelt:      " << res_kobbelt << "\n";
    cout << "Long Acc:     " << res_long << "\n";
    cout << "Pichat:       " << res_pichat << "\n";
    cout << "Sorting:      " << res_sorting << "\n";

    return 0;
}