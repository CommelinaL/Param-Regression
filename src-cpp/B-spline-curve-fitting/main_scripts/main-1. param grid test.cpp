#include <numeric>
#include <Eigen/Dense>
#include <cmath>
#include <Windows.h>
#include <glut.h>
#include <gl/GL.h>
#include <gl/GLU.h>
#include "tinynurbs.h"
#include <cmath>
#include <stdlib.h>
#include "omp.h"
#include "alg_Approximation.h"
#include "geo_Approximation.h"
#include "ExtractFeatures.h"
#include "GenerateData.h"
#include <mutex>
#define OPENGA_EXTERN_LOCAL_VARS
std::mutex mtx_rand;
#include "param_knot_GA.hpp"
#include "vec_file.hpp"
#include "parallel_sort.hpp"
#include "csv_writer.hpp"

using param_func = std::function<void(const std::vector<glm::dvec3>& pts, int degree,
	std::vector<double>& knots, std::vector<double>& param)>;

double eval_param_func(param_func& func, const std::vector<glm::dvec3>& points,
	std::vector<double>& param, tinynurbs::Curve<double>& crv) {
	// Initialization
	crv.knots.clear();
	param.clear();
	crv.control_points.clear();
	crv.degree = 3;
	func(points, 3, crv.knots, param);
	GlobalInterp(points, 3, crv.knots, param, crv.control_points);
	double metric = 0;
	for (int j = 1; j < points.size(); j++)
	{
		metric += calculate2013Criteria(crv, param[j - 1], param[j]);
	}
	metric /= (points.size() - 1);
	return metric;
}

double eval_param(const std::vector<glm::dvec3>& points,
	std::vector<double>& param, tinynurbs::Curve<double>& crv) {
	// Initialization
	int degree = 3;
	int n = points.size();
	int m = n + degree + 1;
	double num = m - 2 * degree - 2;
	crv.knots.resize(m);
	for (int i = 0; i <= degree; i++)crv.knots[i] = 0;
	for (int i = degree + 1; i < degree + num + 1; i++) {
		double sum = 0;
		for (int k = i - degree; k <= i - 1; k++) {
			sum += param[k];
		}
		crv.knots[i] = sum / (double)degree;
	}
	for (int i = degree + num + 1; i < m; i++)crv.knots[i] = 1;
	crv.control_points.clear();
	crv.degree = degree;
	GlobalInterp(points, degree, crv.knots, param, crv.control_points);
	double metric = 0;
	for (int j = 1; j < points.size(); j++)
	{
		metric += calculate2013Criteria(crv, param[j - 1], param[j]);
	}
	metric /= (points.size() - 1);
	return metric;
}


int main(int argc, char** argv) {
	int seq_len = 15;
	std::string model_name = "XGBoost";
	std::string param_name = "n_estimators";
	std::string project_root = PROJECT_ROOT;
	std::string DataDir = project_root + "\\variable_length\\split_dataset_" + std::to_string(seq_len) + "\\test\\"; // The directory where sample is located
	std::string pred_dir = project_root + "\\variable_length\\split_dataset_" + std::to_string(seq_len) + "\\param_grid_test\\" + model_name + "_on_" + param_name + "\\";
	std::string data_dir_tpl = DataDir + "*";
	std::vector<double> method_cost;
	std::vector<std::string> param;
	int total_count = 0;

	WIN32_FIND_DATAA findFileData;
	HANDLE hFind = FindFirstFileA(data_dir_tpl.c_str(), &findFileData);
	if (hFind == INVALID_HANDLE_VALUE) {
		std::cerr << "No files found according to the template: " << data_dir_tpl << std::endl;
		return -1;
	}
	do {
		std::string PointPath = DataDir + findFileData.cFileName + "\\point_data.txt";
		if (std::string(findFileData.cFileName) == "." || std::string(findFileData.cFileName) == "..") continue;
		std::string file_id = findFileData.cFileName;
		total_count++;
		if (total_count > 1000) break;
		if (total_count % 20 == 0) std::cout << "Testing on " << PointPath << std::endl;
		std::ifstream file(PointPath);
		if (file.is_open()) {
			// Read the points
			int point_num;
			file >> point_num;
			std::vector<glm::dvec3> pts;
			for (int i = 0; i < point_num; i++)
			{
				double x, y;
				file >> x >> y;
				pts.push_back(glm::dvec3(x, y, 0.0));
			}
			file.close();
			// Testing regressors
			if (param.empty()) {
				WIN32_FIND_DATAA findParamData;
				std::string param_item_tpl = pred_dir + file_id + "\\*";
				HANDLE hParamFind = FindFirstFileA(param_item_tpl.c_str(), &findParamData);
				if (hParamFind == INVALID_HANDLE_VALUE) {
					std::cerr << "No files found according to the template: " << param_item_tpl << std::endl;
					return -1;
				}
				else {
					do {
						if (std::string(findParamData.cFileName) == "." || std::string(findParamData.cFileName) == ".."
							|| std::string(findParamData.cFileName) == "point_data.txt") continue;
						param.push_back(findParamData.cFileName);
						param[param.size() - 1].erase(param[param.size() - 1].size() - 4);
						method_cost.push_back(0);
					} while (FindNextFileA(hParamFind, &findParamData) != 0);
				}
			}
			for (int j = 0; j < param.size(); j++) {
				std::string pred_path = pred_dir + file_id + "\\" + param[j] + ".bin";
				std::vector<double> intervals;
				readVectorFromFile(intervals, pred_path);
				param_func regress_func = [&](const std::vector<glm::dvec3>& pts, int degree,
					std::vector<double>& knots, std::vector<double>& param)
					{ return RegressorInterp(pts, degree, knots, param, intervals); };
				std::vector<double> param;
				tinynurbs::Curve<double> crv;
				method_cost[j] += eval_param_func(regress_func, pts, param, crv);
			}
		}
		else std::cerr << "Failed to open file: " << PointPath << std::endl;
	} while (FindNextFileA(hFind, &findFileData) != 0);
	FindClose(hFind);
	// Output results
	for (int i = 0; i < method_cost.size(); i++) {
		method_cost[i] /= (double)total_count;
	}
	// Save results
	write_csv("grid_test_" + model_name + "_on_" + param_name + "_.csv", param, method_cost);
	std::cout << "total count: " << total_count;
	return 0;
}
