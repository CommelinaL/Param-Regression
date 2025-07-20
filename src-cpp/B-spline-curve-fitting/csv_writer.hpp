#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include "vec_file.hpp"

template <class Vector>
void write_row(const Vector& vec, const std::string& filename, bool append = false) {
    std::ofstream outFile(filename, append ? std::ios_base::app : std::ios_base::out);
    if (outFile.is_open()) {
        for (size_t i = 0; i < vec.size() - 1; ++i) {
            outFile << vec[i] << ',';
        }
        outFile << vec.back() << '\n';
    }
    else {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }
}

// Base case: write the last vector
template<class Vector>
void write_csv_helper(const std::string& filename, const Vector& vec) {
    write_row(vec, filename, true);
}

// Recursive case: write the current vector and call for the rest
template<class Vector, class... Vectors>
void write_csv_helper(const std::string& filename, const Vector& vec, const Vectors&... vectors) {
    write_row(vec, filename, true);
    write_csv_helper(filename, vectors...);
}

// Main write_csv function
template<class... Vectors>
void write_csv(const std::string& filename, const std::vector<std::string>& titles, const Vectors&... vectors) {
    create_directory_if_not_exists(filename, true);
    write_row(titles, filename); // Write titles first
    write_csv_helper(filename, vectors...); // Write the rest of the vectors
}