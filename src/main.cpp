#include "merge.hpp"
#include "fma.hpp"
#include "kobbelt.hpp"
#include "long_accumulator.hpp"
#include "pichat.hpp"
#include "sorting.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cmath>

using namespace std;

void print_vector_summary(const std::vector<double>& v, const std::string& name)
{
    using std::cout;
    using std::isnan; using std::isinf;

    const size_t n = v.size();
    cout << name << ": ";
    if (n <= 5) {                   
        cout << "[";
        for (size_t j = 0; j < n; ++j) {
            if (j) cout << ", ";
            if (isnan(v[j]))                cout << "NaN";
            else if (isinf(v[j]))           cout << (v[j] > 0 ? "INF" : "-INF");
            else                            cout << std::scientific << v[j];
        }
        cout << "]";
    } else {                          
        cout << "[size = " << n << " elements]";
    }
    cout << '\n';
}


int main(){
    vector<double> a = {1.0, 2.0, 3.0, 1e300, 1e-300};
    vector<double> b = {4.0, 5.0, 6.0, 1e-300, 1e300};

    double merge_fma = merge_dot_product(a, b);
    double res_fma = fma_dot_product(a, b);
    double res_kobbelt = kobbelt_dot_product(a, b);
    double res_long = long_accumulator_dot_product(a, b);
    double res_pichat = pichat_dot_product(a, b);
    double res_sorting = sorting_dot_product(a, b);

    cout << "MERGE:        " << merge_fma << "\n";
    cout << "FMA:          " << res_fma << "\n";
    cout << "Kobbelt:      " << res_kobbelt << "\n";
    cout << "Long Acc:     " << res_long << "\n";
    cout << "Pichat:       " << res_pichat << "\n";
    cout << "Sorting:      " << res_sorting << "\n";

    return 0;
}