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

double eval_curvature(const std::vector<glm::dvec3>& points, const std::vector<double>& param, tinynurbs::Curve<double>& crv) {
	double max_curvature = 0;
	for (int j = 1; j < points.size(); j++) {
		double curvature = calculateMaxCurvatureOnPartCurve(crv, param[j - 1], param[j]);
		if (curvature > max_curvature) {
			max_curvature = curvature;
		}
	}
	return max_curvature;
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
	int dataset_size = 250000;
	std::string del_feat = "npc";
	std::string model_name = "mlp";
	std::string project_root = PROJECT_ROOT;
	std::string DataDir = project_root + "\\variable_length\\split_dataset_" + std::to_string(seq_len) + "\\test\\"; // The directory where sample is located
	std::string pred_dir = project_root + "\\variable_length\\" + model_name + "_wo_" + del_feat + "\\test_local_len_on" + std::to_string(seq_len) + "\\" + std::to_string(dataset_size) + "\\";
	std::string cost_dir = project_root + "\\variable_length\\split_dataset_" + std::to_string(seq_len) + "\\cost\\local_len_size_test\\" + model_name + "_wo_" + del_feat + "\\";
	std::string data_dir_tpl = DataDir + "*";
	std::string other_cost_dir = project_root + "\\variable_length\\cost\\split_dataset_" + std::to_string(seq_len) + "\\test\\";
	std::vector<int> top_cnt;
	std::vector<int> top_3_cnt;
	std::vector<double> method_cost;
	std::vector<double> cost_wo_cusp;
	std::vector<double> cost_cusp;
	std::vector<int> cusp_cnt;
	std::vector<double> worst_cost;
	std::vector<double> sec_worst;
	std::vector<double> thr_worst;
	std::vector<double> cost_wo_worst;
	std::vector<double> cost_wo_worst_3;
	std::vector<std::string> local_len;
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
			// Testing heuristic methods
			std::vector<std::string> heuristic_name = { "uniform", "chord", "centripetal", "universal",
				"foley", "fang", "modified_chord", "zcm" };
			std::vector<param_func> method_func = { UniformInterp, ChordInterp, CentripetalInterp, UniversalInterp,
				CorrectChordInterp,	RefinedCentripetalInterp, ModifiedChordInterp, ZCMInterp };
			std::vector<double> heuristic_cost(heuristic_name.size(), 0);
			for (int i = 0; i < 8; i++) {
				std::string cost_path = other_cost_dir + heuristic_name[i] + "\\" + file_id + ".bin";
				double cost_i = 0;
				std::ifstream in_cost_file(cost_path, std::ios::binary);
				if (in_cost_file.is_open()) {
					in_cost_file.read(reinterpret_cast<char*>(&cost_i), sizeof(double));
					in_cost_file.close();
				}
				else {
					std::vector<double> param;
					tinynurbs::Curve<double> crv;
					cost_i = eval_param_func(method_func[i], pts, param, crv);
					create_directory_if_not_exists(cost_path, true);
					std::ofstream out_cost_file(cost_path, std::ios::binary);
					out_cost_file.write(reinterpret_cast<const char*>(&cost_i), sizeof(double));
					out_cost_file.close();
				}
				heuristic_cost[i] = cost_i;
			}
			std::sort(heuristic_cost.begin(), heuristic_cost.end());
			// Testing regressors
			if (local_len.empty()) {
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
						std::string tmp = findParamData.cFileName;
						tmp.erase(tmp.size() - 4);
						top_cnt.push_back(0);
						top_3_cnt.push_back(0);
						local_len.push_back(tmp);
						method_cost.push_back(0);
						cost_wo_cusp.push_back(0);
						cost_cusp.push_back(0);
						cusp_cnt.push_back(0);
						worst_cost.push_back(0);
						sec_worst.push_back(0);
						thr_worst.push_back(0);
						cost_wo_worst.push_back(0);
						cost_wo_worst_3.push_back(0);
					} while (FindNextFileA(hParamFind, &findParamData) != 0);
				}
			}
			for (int j = 0; j < local_len.size(); j++) {
				std::string curvature_path = cost_dir + local_len[j] + "\\" + file_id + "_crv.bin";
				std::string cost_path = cost_dir + local_len[j] + "\\" + file_id + ".bin";
				std::string pred_path = pred_dir + file_id + "\\" + local_len[j] + ".bin";
				std::vector<double> intervals;
				readVectorFromFile(intervals, pred_path);
				double cost_j = 0, curvature_j = 0;
				std::ifstream in_crv_file(curvature_path, std::ios::binary);
				if (in_crv_file.is_open()) {
					in_crv_file.read(reinterpret_cast<char*>(&curvature_j), sizeof(double));
					in_crv_file.close();
					std::ifstream in_cost_file(cost_path, std::ios::binary);
					in_cost_file.read(reinterpret_cast<char*>(&cost_j), sizeof(double));
					in_cost_file.close();
				}
				else {
					param_func regress_func = [&](const std::vector<glm::dvec3>& pts, int degree,
						std::vector<double>& knots, std::vector<double>& param)
						{ return VarRegressorInterp(pts, degree, knots, param, intervals, std::stoi(local_len[j])); };
					std::vector<double> param;
					tinynurbs::Curve<double> crv;
					cost_j = eval_param_func(regress_func, pts, param, crv);
					curvature_j = eval_curvature(pts, param, crv);
					create_directory_if_not_exists(cost_path, true);
					std::ofstream out_cost_file(cost_path, std::ios::binary);
					out_cost_file.write(reinterpret_cast<const char*>(&cost_j), sizeof(double));
					out_cost_file.close();
					std::ofstream out_crv_file(curvature_path, std::ios::binary);
					out_crv_file.write(reinterpret_cast<const char*>(&curvature_j), sizeof(double));
					out_crv_file.close();
					if (std::stoi(local_len[j]) == 4) {
						param_func regress_4_func = [&](const std::vector<glm::dvec3>& pts, int degree,
							std::vector<double>& knots, std::vector<double>& param)
							{ return RegressorInterp(pts, degree, knots, param, intervals); };
						std::vector<double> param_4;
						tinynurbs::Curve<double> crv_4;
						eval_param_func(regress_4_func, pts, param_4, crv_4);
						for (int k = 0; k < param.size(); k++) {
							assert(std::abs(param[k] - param_4[k]) < 1e-6);
						}
					}
					if (std::stoi(local_len[j]) == seq_len) {
						for (int k = 1; k < param.size(); k++) {
							assert(std::abs(param[k] - param[k - 1] - intervals[k - 1]) < 1e-6);
						}
					}
				}
				method_cost[j] += cost_j;
				if (curvature_j > 1e4) {
					cost_cusp[j] += cost_j;
					cusp_cnt[j]++;
				}
				if (cost_j < heuristic_cost[0]) {
					top_cnt[j]++;
				}
				if (cost_j < heuristic_cost[2]) {
					top_3_cnt[j]++;
				}
				if (cost_j > worst_cost[j]) {
					thr_worst[j] = sec_worst[j];
					sec_worst[j] = worst_cost[j];
					worst_cost[j] = cost_j;
				}
				else if (cost_j > sec_worst[j]) {
					thr_worst[j] = sec_worst[j];
					sec_worst[j] = cost_j;
				}
				else if (cost_j > thr_worst[j]) {
					thr_worst[j] = cost_j;
				}
			}
		}
		else std::cerr << "Failed to open file: " << PointPath << std::endl;
	} while (FindNextFileA(hFind, &findFileData) != 0);
	FindClose(hFind);
	// Output results
	for (int i = 0; i < method_cost.size(); i++) {
		cost_wo_cusp[i] = (method_cost[i] - cost_cusp[i]) / (double)(total_count - 1 - cusp_cnt[i]);
		cost_wo_worst_3[i] = (method_cost[i] - worst_cost[i] - sec_worst[i] - thr_worst[i]) / (double)(total_count - 4);
		cost_wo_worst[i] = (method_cost[i] - worst_cost[i]) / (double)(total_count - 2);
		method_cost[i] /= (double)(total_count - 1);
	}
	// Save results
	write_csv("local_len_" + model_name + "_wo_" + del_feat + "_.csv", local_len,
		top_cnt, top_3_cnt, cost_wo_cusp, cusp_cnt, cost_wo_worst_3, cost_wo_worst, method_cost);
	std::cout << "total count: " << total_count;
	return 0;
}