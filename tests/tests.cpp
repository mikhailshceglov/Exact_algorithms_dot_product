#include "fma.hpp"
#include "kobbelt.hpp"
#include "long_accumulator.hpp"
#include "pichat.hpp"
#include "sorting.hpp"
#include <gmp.h>
#include <gmpxx.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <cstring>
#include <cfenv>
#include <tuple>

using namespace std;

#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"

struct AlgorithmResult{
    string name;
    double result;
    bool gmp_valid;
    bool permutations_ok;
};

// Точное скалярное произведение через GMP
double ExactDotProductGMP(const vector<double>& a, const vector<double>& b){
    // Проверка специальных случаев
    bool has_nan = false;
    int pos_infs = 0, neg_infs = 0;
    for (int i = 0; i < a.size(); i++){
        if (isnan(a[i]) || isnan(b[i])){
            has_nan = true;
            break;
        }
        if (isinf(a[i])) (a[i] > 0) ? ++pos_infs : ++neg_infs;
        if (isinf(b[i])) (b[i] > 0) ? ++pos_infs : ++neg_infs;
    }

    if (has_nan) return numeric_limits<double>::quiet_NaN();
    if (pos_infs && neg_infs) return numeric_limits<double>::quiet_NaN();
    if (pos_infs) return numeric_limits<double>::infinity();
    if (neg_infs) return -numeric_limits<double>::infinity();

    // Точное вычисление с GMP
    mpf_t sum, tmp_a, tmp_b, product;
    mpf_init2(sum, 512);
    mpf_init2(tmp_a, 512);
    mpf_init2(tmp_b, 512);
    mpf_init2(product, 512);
    mpf_set_d(sum, 0.0);

    for (int i = 0; i < a.size(); i++){
        mpf_set_d(tmp_a, a[i]);
        mpf_set_d(tmp_b, b[i]);
        mpf_mul(product, tmp_a, tmp_b);  // Умножение a[i]*b[i]
        mpf_add(sum, sum, product);      // Суммирование
    }

    double result = mpf_get_d(sum);
    mpf_clear(sum);
    mpf_clear(tmp_a);
    mpf_clear(tmp_b);
    mpf_clear(product);
    return result;
}

// Проверка инвариантности к перестановкам
template<double DotProductFunc(const vector<double>&, const vector<double>&)>
bool CheckPermutations(const vector<double>& a, const vector<double>& b){
    if (a.size() != b.size()) return false;
    if (a.size() < 2) return true;
    
    const double original = DotProductFunc(a, b);
    if (isnan(original)) return true;
    
    vector<size_t> indices(a.size());
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), mt19937{random_device{}()});
    
    vector<double> a_shuffled(a.size());
    vector<double> b_shuffled(b.size());
    for (int i = 0; i < a.size(); i++){
        a_shuffled[i] = a[indices[i]];
        b_shuffled[i] = b[indices[i]];
    }
    
    const double permuted = DotProductFunc(a_shuffled, b_shuffled);
    return original == permuted;
}

int main() {
    fedisableexcept(FE_ALL_EXCEPT);
    
    struct TestCase {
        vector<double> a;
        vector<double> b;
        string description;
    };

    vector<TestCase> tests = {
        {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, "Простое скалярное произведение"},
        {{1e100, 1.0, -1e100}, {1.0, 1.0, 1.0}, "Компенсация больших чисел"},
        {{1e30, 1.0, -1e30}, {1.0, 1.0, 1.0}, "Разные порядки величин"},
        {{1.0, 1e100, 1.0, -1e100}, {1.0, 1.0, 1.0, 1.0}, "Сложная компенсация"},
        {{DBL_EPSILON, DBL_EPSILON}, {1.0, 1.0}, "Сумма малых чисел"},
        {{1.0, -1.0, 1.0, -1.0}, {1.0, 1.0, 1.0, 1.0}, "Чередующиеся знаки"},
        {{NAN, 2.0}, {1.0, 1.0}, "NaN в данных"},
        {{INFINITY, 1.0}, {1.0, 1.0}, "Бесконечности"},
        {{INFINITY, -INFINITY}, {1.0, 1.0}, "Противоположные бесконечности"},
        {{DBL_MIN / 2, DBL_MIN / 2}, {1.0, 1.0}, "Денормализованные числа"}
    };

    vector<tuple<string, double(*)(const vector<double>&, const vector<double>&), bool(*)(const vector<double>&, const vector<double>&)>> algorithms = {
        {"FMA", fma_dot_product, CheckPermutations<fma_dot_product>},
        {"Kobbelt", kobbelt_dot_product, CheckPermutations<kobbelt_dot_product>},
        {"Long Acc", long_accumulator_dot_product, CheckPermutations<long_accumulator_dot_product>},
        {"Pichat", pichat_dot_product, CheckPermutations<pichat_dot_product>},
        {"Sorting", sorting_dot_product, CheckPermutations<sorting_dot_product>}
    };

    for (int i = 0; i < tests.size(); i++){
        const auto& test = tests[i];
        
        cout << BLUE << "══════════════════════════════════════════════════\n"
             << "Тест #" << i + 1 << ": " << test.description << RESET << "\n"
             << "Вектор A: [";
        
        for (size_t j = 0; j < test.a.size(); ++j) {
            if (j > 0) cout << ", ";
            if (isnan(test.a[j])) cout << "NaN";
            else if (isinf(test.a[j])) cout << ((test.a[j] > 0) ? "INF" : "-INF");
            else cout << scientific << test.a[j];
        }
        cout << "]\nВектор B: [";
        
        for (size_t j = 0; j < test.b.size(); ++j) {
            if (j > 0) cout << ", ";
            if (isnan(test.b[j])) cout << "NaN";
            else if (isinf(test.b[j])) cout << ((test.b[j] > 0) ? "INF" : "-INF");
            else cout << scientific << test.b[j];
        }
        cout << "]" << "\n";

        // Точное значение через GMP
        double gmp_exact = ExactDotProductGMP(test.a, test.b);
        cout << YELLOW << "Точное значение (GMP): ";
        if (isnan(gmp_exact)) cout << "NaN";
        else if (isinf(gmp_exact)) cout << ((gmp_exact > 0) ? "INF" : "-INF");
        else cout << scientific << setprecision(15) << gmp_exact;
        cout << RESET << "\n\n";

        // Запуск алгоритмов
        for (const auto& algo : algorithms) {
            feclearexcept(FE_ALL_EXCEPT);
            
            AlgorithmResult res;
            res.name = get<0>(algo);
            res.result = get<1>(algo)(test.a, test.b);
            
            // Проверка совпадения с GMP
            if (isnan(res.result)) {
                res.gmp_valid = isnan(gmp_exact);
            } 
            else if (isinf(res.result)) {
                res.gmp_valid = isinf(gmp_exact) && (signbit(res.result) == signbit(gmp_exact));
            } 
            else {
                res.gmp_valid = (memcmp(&res.result, &gmp_exact, sizeof(double)) == 0);
            }
            
            res.permutations_ok = get<2>(algo)(test.a, test.b);

            cout << " - Алгоритм: " << left << setw(10) << res.name 
                 << " Результат: " << scientific << setprecision(15) << res.result
                 << " GMP: " << (res.gmp_valid ? GREEN "✓" : RED "✗") << RESET
                 << " Перестановки: " << (res.permutations_ok ? GREEN "✓" : RED "✗") << RESET
                 << "\n";
        }
        cout << BLUE << "══════════════════════════════════════════════════\n\n" << RESET;
    }

    return 0;
}