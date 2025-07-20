#include "vec_file.hpp"
#include <string.h>
#include <wchar.h>

bool create_directory_if_not_exists(const std::string& path, bool createParent, bool silent) {
    std::string directoryPath;

    // Determine the target path based on the argument
    if (createParent) {
        // Extract the parent directory
        size_t lastSeparator = path.find_last_of("\\/");
        if (lastSeparator != std::string::npos) {
            directoryPath = path.substr(0, lastSeparator);
        } else {
            // No separator means no parent directory to create
            return false;
        }
    } else {
        // Use the path as it is (the directory itself)
        directoryPath = path;
    }

    // Check if the directoryPath exists
    DWORD dwAttrib = GetFileAttributesA(directoryPath.c_str());
    if (dwAttrib == INVALID_FILE_ATTRIBUTES) {
        // Split the path into components and create them one by one
        std::string tempPath;
        for (size_t i = 0; i < directoryPath.length(); ++i) {
            // If we find a separator, try to create the directory
            if (directoryPath[i] == '\\' || directoryPath[i] == '/') {
                // Create the directory for the current segment
                if (!tempPath.empty()) {
                    tempPath += '\\';  // Ensure to use the correct separator
                    if ((CreateDirectoryA(tempPath.c_str(), NULL) || GetLastError() == ERROR_ALREADY_EXISTS) && silent == false) {
                        std::cout << "Created directory: " << tempPath << std::endl;
                    }
                }
            } else {
                tempPath += directoryPath[i];
            }
        }

        // Finally, create the last directory in the path
        if (!tempPath.empty()) {
            if ((CreateDirectoryA(tempPath.c_str(), NULL) || GetLastError() == ERROR_ALREADY_EXISTS) && silent == false) {
                std::cout << "Created directory: " << tempPath << std::endl;
            }
        }
        return true;
    }
    
    return (dwAttrib & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

bool copyFile(const std::string& sourcePath, const std::string& destinationPath) {
    if (sourcePath == destinationPath)
        return true;
    std::ifstream sourceFile(sourcePath, std::ios::binary);
    if (!sourceFile.is_open()) {
        std::cerr << "Unable to open source file: " << sourcePath << std::endl;
        return false;
    }

    std::ofstream destinationFile(destinationPath, std::ios::binary);
    if (!destinationFile.is_open()) {
        char errorMsg[256]; // Buffer for the error message
        strerror_s(errorMsg, sizeof(errorMsg), errno); // Get the error message
        std::cerr << "Unable to open destination file: " << destinationPath
            << " (Error: " << errorMsg << ")" << std::endl; 
        return false;
    }

    destinationFile << sourceFile.rdbuf();

    sourceFile.close();
    destinationFile.close();
    return true;
}