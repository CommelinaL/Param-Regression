# C++ Requirements

To build the C++ components, set up your development environment as follows:

## Development Environment Setup

### 1. Windows SDK Version 10.0
- Open **Visual Studio Installer**
- Under **Workloads**, select **Desktop development with C++**
- In **Installation details**, ensure **Windows 10 SDK (10.0.xxxx)** is checked (any recent version)
- Click **Modify** to install if needed

### 2. Platform Toolset: Visual Studio 2022 (v143)
- In **Visual Studio Installer**, verify under **Desktop development with C++** that **MSVC v143 - VS 2022 C++ build tools** is selected
- Click **Modify** if not installed

### 3. C++ Language Standard: ISO C++14 (`/std:c++14`)
- The project uses C++14 standard; no additional setup needed if using Visual Studio 2022

## Project Configuration

### 4. Target Platform: Win32 (32-bit)
- Ensure you're building for the correct architecture (32-bit as specified in project settings)
- Select **Win32** platform in the configuration dropdown, not x64

### 5. Configuration Type: Application (.exe)
- This project builds an executable application
- Output will be a standalone `.exe` file (`src-cpp\Release\B-spline-curve-fitting.exe`)

### 6. MSVC Runtime: Default (Legacy MSVC)
- Project uses the default legacy MSVC runtime configuration
- No additional runtime setup required

## Path Configuration

### 7. Include and Library Directories
- In **Visual Studio**, go to **Project Properties** > **Configuration Properties** > **VC++ Directories**
- Replace any `D:\BSplineLearning` paths in **Include Directories** and **Library Directories** with your own `PROJECT_ROOT` path
- Ensure all referenced library paths point to your local development environment

## Build Instructions

1. Open the solution file in Visual Studio 2022
2. Select **Active(Release)** configuration and **Active(Win32)** platform
3. Build the solution using **Build** > **Build Solution** or `Ctrl+Shift+B`

## Troubleshooting

- If you encounter SDK version conflicts, ensure the Windows SDK version matches what's installed on your system
- For path-related errors, double-check that all directory references have been updated from the original `D:\BSplineLearning` paths
- If building fails due to C++ standard issues, verify that your Visual Studio 2022 installation includes C++14 support