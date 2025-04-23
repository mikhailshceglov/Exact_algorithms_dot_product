#include "merge.hpp"
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

void print_vector_summary(const std::vector<double>& v, const std::string& name);


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

int main() {
    fedisableexcept(FE_ALL_EXCEPT);
    
    struct TestCase {
        vector<double> a;
        vector<double> b;
        string description;
    };

    /*
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
    */

    mt19937_64 rng(20250423);                       // фиксируем сид
    uniform_real_distribution<> uni01(0.0, 1.0);    // для мантиссы
    uniform_int_distribution<int> sign01(0, 1);     // случайный знак

    auto rnd_pow2 = [&](int exp_min, int exp_max) {
        int e = uniform_int_distribution<int>(exp_min, exp_max)(rng);
        double m = ldexp(uni01(rng) + 1.0, -1);     // мантисса ∈ [0.5,1)
        return copysign(ldexp(m, e), sign01(rng) ? 1.0 : -1.0);
    };

    vector<TestCase> tests;

    // 1. Катастрофическая компенсация 
    {
        size_t n = 10000;
        vector<double> a(n), b(n);
        for (size_t i = 0; i < n; ++i) {
            double x = (i % 2 == 0 ? 1e308 : -1e308);
            a[i] = x;
            b[i] = 1.0;
        }
        // маленький хвост
        a.push_back(1e-10); b.push_back(1e-10);
        tests.push_back({move(a), move(b),
            "Катастрофическая компенсация (±1e308 * 1 + хвост 1e-10)"});
    }

    
    // 2. Разреженные экспоненты и субнормали 
    {
        size_t n = 5000;
        vector<double> a(n), b(n);
        for (size_t i = 0; i < n; ++i) {
            a[i] = rnd_pow2(-1022, 1023);           // обычные числа
            b[i] = rnd_pow2(-1074, -1022);          // субнормали
        }
        tests.push_back({move(a), move(b),
            "Сильная разреженность экспонент + субнормали"});
    }
    

    // 3. NaN / Inf комбинации 
    tests.push_back({
        {numeric_limits<double>::infinity(),
        0.0,
        -numeric_limits<double>::infinity(),
        1.0},
        {0.0,
        numeric_limits<double>::infinity(),
        1.0,
        1.0},
        "Спецзначения (Inf·0 = NaN, Inf + (–Inf) → NaN)"
    });

    // 4. Перестановочная ловушка (шахматка знаков) 
    {
        size_t n = 10001;                          // нечётно → точная сумма ноль
        vector<double> a(n), b(n);
        for (size_t i = 0; i < n; ++i) {
            a[i] = (i % 2 ? 1.0 : -1.0) * ldexp(1.0, i % 32 - 16); // ±2^[-16..15]
            b[i] = 1.0;
        }
        tests.push_back({move(a), move(b),
            "Перестановочная ловушка (чередование знаков и экспонент)"});
    }

    
    // 5. Большие случайные массивы 
    {
        const size_t n = 100000;
        vector<double> a(n), b(n);
        for (size_t i = 0; i < n; ++i) {
            a[i] = rnd_pow2(-1022, 1023);
            b[i] = rnd_pow2(-1022, 1023);
        }
        tests.push_back({move(a), move(b),
            "Случайный большой тест (100k элементов, равномерный log-распр.)"});
    }
    

    // 6. Переполнение в отдельном произведении, конечная сумма 
    {
        vector<double> a = {1e308,  1e308, -1e308};
        vector<double> b = {1e308, -1e308,  0.0   };   // 0 · 1e308 = 0
        tests.push_back({move(a), move(b),
            "Переполнение в произведении, но конечная сумма"});
    }

    // EXTRA TEST 7
    {
        const size_t n = 100000;
        const double eps = std::ldexp(1.0, -54);   // ε = 2^-54  (непредставимо!)
        std::vector<double> a(n, 1.0 + eps);
        std::vector<double> b(n, 1.0 + eps);

        tests.push_back({std::move(a), std::move(b),
            "Систематическое смещение (1+ε)*(1+ε) × 100k"}); // ≈ 2e-10 ошибка
    }  

    // EXTRA TEST 8
    {
        const size_t n = 2000000;                 // 2 млн
        const double big = 1.0e150;               // ≈ 2^498
        const double tiny = 1.0e-150;             // ≈ 2^-498

        std::vector<double> a, b;
        a.reserve(n); b.reserve(n);

        // первая половина  (+big)*(+tiny)  →  ~+1
        for (size_t i = 0; i < n/2; ++i) { a.push_back( big); b.push_back( tiny); }

        // вторая половина (−big)*(−tiny)  →  ~+1  (тот же знак!)
        // потом поменяем порядок (shuffle проверка)
        for (size_t i = 0; i < n/2; ++i) { a.push_back(-big); b.push_back(-tiny); }

        // Точный ответ ≈ n  (но молодшие ~20-30 бит зависят от хвостов)
        tests.push_back({std::move(a), std::move(b),
            "Сильная компенсация big·tiny (2 млн элементов)"});
    }
    

    // EXTRA TEST 9
    {
        using std::numeric_limits;
        const double sub = numeric_limits<double>::min() * 0.25; // ~5.56e-309

        const size_t n = 500000;      // 5×10^5 ⇒ итог ≈ 2.78e-303 (не нуль!)
        std::vector<double> a(n, sub);
        std::vector<double> b(n, 1.0);

        // Чтобы показать контраст, добавим две «большие» строки,
        // дающие точный ноль:  +1 −1.
        a.push_back( 1.0); b.push_back( 1.0);
        a.push_back(-1.0); b.push_back( 1.0);

        tests.push_back({std::move(a), std::move(b),
            "Субнормальный дождь + точное взаимное обнуление"});
    }

    vector<tuple<string, double(*)(const vector<double>&, const vector<double>&), bool(*)(const vector<double>&, const vector<double>&)>> algorithms = {
        {"Merge", merge_dot_product, CheckPermutations<merge_dot_product>},
        {"FMA", fma_dot_product, CheckPermutations<fma_dot_product>},
        //{"Kobbelt", kobbelt_dot_product, CheckPermutations<kobbelt_dot_product>},
        {"Long Acc", long_accumulator_dot_product, CheckPermutations<long_accumulator_dot_product>},
        //{"Pichat", pichat_dot_product, CheckPermutations<pichat_dot_product>},
        {"Sorting", sorting_dot_product, CheckPermutations<sorting_dot_product>}
    };

    for (size_t t = 0; t < tests.size(); ++t) {
        const auto& test = tests[t];
        const int   num  = static_cast<int>(t + 1);      // «человеческий» номер

        /* ── ШАПКА ────────────────────────────────────────────────────────────── */
        std::cout << BLUE
                << "══════════════════════════════════════════════════\n"
                << "Тест #" << num << ": " << test.description << RESET << '\n';

        print_vector_summary(test.a, "Вектор A");
        print_vector_summary(test.b, "Вектор B");

        /* ── GMP reference ───────────────────────────────────────────────────── */
        double gmp_exact = ExactDotProductGMP(test.a, test.b);
        std::cout << YELLOW << "GMP: ";
        if (std::isnan(gmp_exact))         std::cout << "NaN";
        else if (std::isinf(gmp_exact))    std::cout << (gmp_exact > 0 ? "INF" : "-INF");
        else                               std::cout << std::scientific << std::setprecision(15)
                                                    << gmp_exact;
        std::cout << RESET << "\n\n";

        /* ── ПРОГОН АЛГОРИТМОВ ───────────────────────────────────────────────── */
        for (const auto& algo : algorithms) {
            std::feclearexcept(FE_ALL_EXCEPT);

            const std::string& name   = std::get<0>(algo);
            double result             = std::get<1>(algo)(test.a, test.b);

            bool gmp_ok;
            if (std::isnan(result))
                gmp_ok = std::isnan(gmp_exact);
            else if (std::isinf(result))
                gmp_ok = std::isinf(gmp_exact) && (std::signbit(result) == std::signbit(gmp_exact));
            else
                gmp_ok = std::memcmp(&result, &gmp_exact, sizeof(double)) == 0;

            bool perm_ok = std::get<2>(algo)(test.a, test.b);

            /* —— ВЫВОД: № теста, алгоритм, результат, статусы —— */
            std::cout << " #"
                    << std::setw(2) << num << "  "                        // номер теста
                    << std::left  << std::setw(10) << name                // имя алгоритма
                    << std::right << std::scientific << std::setprecision(15)
                    << result
                    << "  GMP "  << (gmp_ok  ? GREEN "✓" : RED "✗") << RESET
                    << "  Perm " << (perm_ok ? GREEN "✓" : RED "✗") << RESET
                    << '\n';
        }

        std::cout << BLUE
                << "══════════════════════════════════════════════════\n\n"
                << RESET;
    }


    return 0;
}