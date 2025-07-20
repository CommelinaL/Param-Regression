#pragma once
#define _USE_MATH_DEFINES
#include <math.h>
#include <string>
#include <fstream>
#include <glm/glm.hpp>
#include <iostream>
#include <vector>
using namespace std;

std::string NumToStr(int num);

double TransferAngle(double angle);

void GenerateDataDirect(int dataNum, int pNum, int flag_num, string DataDir = "PointData");

bool GenerateTrainData(string inputFileName, int num, string outputFileName, vector<int>& TrainTypeNum, vector<int>& TestTypeNum, int train_id_num, int test_id_num, int pNum);