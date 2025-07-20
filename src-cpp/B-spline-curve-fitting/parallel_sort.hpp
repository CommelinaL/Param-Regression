#include <vector>
#include <algorithm>
#include <numeric>
#include <utility>

using iter = std::vector<int>::iterator;
using order_type = std::vector<std::pair<size_t, iter>>;

template<typename T>
void reorder(std::vector<T>& vect, const std::vector<size_t>& order)
{
    std::vector<bool> done(vect.size(), false);
    for (size_t i = 0; i < vect.size(); ++i)
    {
        if (!done[i])
        {
            done[i] = true;
            size_t current = i;
            T temp = std::move(vect[i]);
            while (order[current] != i)
            {
                vect[current] = std::move(vect[order[current]]);
                current = order[current];
                done[current] = true;
            }
            vect[current] = std::move(temp);
        }
    }
}

// Base case: reorder the last vector
template<class Vector>
void parallel_sort_helper(const std::vector<size_t>& order, Vector& vec) {
    reorder(vec, order);
}

// Recursive case: reorder the current vector and call for the rest
template<class Vector, class... Vectors>
void parallel_sort_helper(const std::vector<size_t>& order, Vector& vec, Vectors&... vectors) {
    reorder(vec, order);
    parallel_sort_helper(order, vectors...);
}

template<class Vector, class... Vectors>
void parallel_sort(Vector& keyvector, Vectors&... vectors) {
    std::vector<size_t> order(keyvector.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return keyvector[a] < keyvector[b];
        });

    parallel_sort_helper(order, keyvector, vectors...);
}
