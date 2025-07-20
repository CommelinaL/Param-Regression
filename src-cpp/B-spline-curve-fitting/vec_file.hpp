#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <Windows.h>

bool create_directory_if_not_exists(const std::string& path, bool createParent = false, bool silent = true);

template<typename T>
void writeVectorToFile(const std::vector<T>& vec, const std::string& filename) {
    create_directory_if_not_exists(filename, true);
    std::ofstream outFile(filename, std::ios::binary); // Open file in binary mode
    if (!outFile) {
        std::cerr << "Error opening file for writing. (" + filename + ")\n";
        return;
    }

    // Write the size of the vector
    std::size_t size = vec.size();
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));

    // Write the vector data
    outFile.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(T));

    outFile.close();
}

template<typename T>
void readVectorFromFile(std::vector<T>& vec, const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary); // Open file in binary mode
    if (!inFile) {
        std::cerr << "Error opening file for reading. (" + filename + ")\n";
        return;
    }

    // Read the size of the vector
    std::size_t size = 0;
    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));

    // Resize the vector to the correct size
    vec.resize(size);

    // Read the vector data
    inFile.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));

    inFile.close();
}

bool copyFile(const std::string& sourcePath, const std::string& destinationPath);