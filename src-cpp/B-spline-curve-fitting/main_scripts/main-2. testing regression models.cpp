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

int main(int argc, char** argv) {
	int point_num = 15;
	std::string DataDir = "D:\\BSplineLearning\\variable_length\\split_dataset_" + std::to_string(point_num) + "\\test\\"; // The directory where sample is located
	int dataset_size = 250000;
	std::string del_feat = "npc";
	std::string PredDir = "D:\\BSplineLearning\\variable_length\\split_dataset_" + std::to_string(point_num) + "\\model_test\\" + std::to_string(dataset_size) + "_wo_" + del_feat + "\\";
	std::string CostDir = "D:\\BSplineLearning\\variable_length\\cost\\split_dataset_" + std::to_string(point_num) + "\\test\\";
	std::string NewCostDir = CostDir + std::to_string(dataset_size) + "_wo_" + del_feat + "\\";
	create_directory_if_not_exists(CostDir);
	create_directory_if_not_exists(NewCostDir);
	std::string data_dir_tpl = DataDir + "*";
	std::vector<std::string> method_name = {
	"Random Forest", "MLP with manual feature", "XGBoost", "Cubic Regression", "Gradient Boosting",
	"GAM", "Quadratic Regression", "Decision Tree", "Ridge Regression", "Linear Regression",
	"SVR", "PLS", "AdaBoost", "Lasso Regression", "ElasticNet", "Bayesian Regression", "RANSAC" }; // Regressors
	std::vector<double> method_cost(method_name.size(), 0);
	std::vector<double> top_rank_cnt(method_name.size(), 0);
	std::vector<double> top_3_rank_cnt(method_name.size(), 0);
	std::vector<double> cost_wo_cusp(method_name.size(), 0);
	std::vector<double> cost_cusp(method_name.size(), 0);
	std::vector<int> cusp_cnt(method_name.size(), 0);
	std::vector<double> worst_cost(method_name.size(), 0);
	std::vector<double> sec_worst(method_name.size(), 0);
	std::vector<double> thr_worst(method_name.size(), 0);
	std::vector<double> cost_wo_worst(method_name.size(), 0);
	std::vector<double> cost_wo_worst_3(method_name.size(), 0);
	std::vector<std::vector<int>> rank_count;
	rank_count.clear();
	for (int i = 0; i < method_name.size(); i++) {
		std::vector<int> tmp(method_name.size(), 0);
		rank_count.push_back(tmp);
	}
	int total_count = 0;

	WIN32_FIND_DATAA findFileData;
	HANDLE hFind = FindFirstFileA(data_dir_tpl.c_str(), &findFileData);
	if (hFind == INVALID_HANDLE_VALUE) {
		std::cerr << "No files found according to the template: " << data_dir_tpl << std::endl;
		return -1;
	}
	do {
		std::string PointPath = DataDir + findFileData.cFileName + "\\point_data.txt";
		std::string file_id = findFileData.cFileName;
		if (std::string(findFileData.cFileName) == "." || std::string(findFileData.cFileName) == "..") continue;
		int id;
		try {
			id = std::stoi(file_id);
		}
		catch (const std::invalid_argument& e) {
			std::cerr << "Invalid argument: " << e.what() << " " << file_id << std::endl;
			continue;
		}
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
			// Parametrization with heuristic methods
			std::vector<std::pair<double, int>> method_rankings(method_name.size());
			// Testing heuristic methods
			std::vector<std::string> heuristic_name = { "uniform", "chord", "centripetal", "universal",
				"foley", "fang", "modified_chord", "zcm" };
			std::vector<param_func> method_func = { UniformInterp, ChordInterp, CentripetalInterp, UniversalInterp,
				CorrectChordInterp,	RefinedCentripetalInterp, ModifiedChordInterp, ZCMInterp };
			std::vector<double> heuristic_cost(heuristic_name.size(), 0);
			for (int i = 0; i < 8; i++) {
				std::string cost_path = CostDir + heuristic_name[i] + "\\" + file_id + ".bin";
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
			for (int i = 0; i < method_name.size(); i++) {
				if (method_name[i] == "Random Forest") continue;
				std::string cost_path = NewCostDir + method_name[i] + "\\" + file_id + ".bin";
				std::string curvature_path = NewCostDir + method_name[i] + "\\" + file_id + "_crv.bin";
				double cost_i = 0, curvature_i = 0;
				std::ifstream in_curvature_file(curvature_path, std::ios::binary);
				if (in_curvature_file.is_open()) {
					in_curvature_file.read(reinterpret_cast<char*>(&curvature_i), sizeof(double));
					in_curvature_file.close();
					std::ifstream in_cost_file(cost_path, std::ios::binary);
					in_cost_file.read(reinterpret_cast<char*>(&cost_i), sizeof(double));
					in_cost_file.close();
				}
				else {
					std::string pred_path = PredDir + file_id + "\\" + method_name[i] + ".bin";
					std::vector<double> intervals;
					readVectorFromFile(intervals, pred_path);
					param_func regress_func = [&](const std::vector<glm::dvec3>& pts, int degree,
						std::vector<double>& knots, std::vector<double>& param)
						{ return RegressorInterp(pts, degree, knots, param, intervals); };
					std::vector<double> param;
					tinynurbs::Curve<double> crv;
					cost_i = eval_param_func(regress_func, pts, param, crv);
					curvature_i = eval_curvature(pts, param, crv);
					create_directory_if_not_exists(cost_path, true);
					std::ofstream out_cost_file(cost_path, std::ios::binary);
					out_cost_file.write(reinterpret_cast<const char*>(&cost_i), sizeof(double));
					out_cost_file.close();
					std::ofstream out_curvature_file(curvature_path, std::ios::binary);
					out_curvature_file.write(reinterpret_cast<const char*>(&curvature_i), sizeof(double));
					out_curvature_file.close();
				}
				assert(cost_i > std::numeric_limits<double>::epsilon());
				method_cost[i] += cost_i;
				method_rankings[i] = { cost_i, i };
				if (cost_i < heuristic_cost[0]) {
					top_rank_cnt[i]++;
				}
				if (cost_i < heuristic_cost[2]) {
					top_3_rank_cnt[i]++;
				}
				if (curvature_i > 1e4) {
					cusp_cnt[i]++;
					cost_cusp[i] += cost_i;
				}
				if (cost_i > worst_cost[i]) {
					thr_worst[i] = sec_worst[i];
					sec_worst[i] = worst_cost[i];
					worst_cost[i] = cost_i;
				}
				else if (cost_i > sec_worst[i]) {
					thr_worst[i] = sec_worst[i];
					sec_worst[i] = cost_i;
				}
				else if (cost_i > thr_worst[i]) {
					thr_worst[i] = cost_i;
				}
			}

			// Sort the methods based on their costs
			std::sort(method_rankings.begin(), method_rankings.end());

			// Update the counts based on the sorted rankings
			for (int rank = 0; rank < method_name.size(); rank++) {
				int method_idx = method_rankings[rank].second;
				rank_count[rank][method_idx]++;
				int crt_rank = rank;
				while (rank + 1 < method_name.size() && method_rankings[rank + 1].first - method_rankings[crt_rank].first < numeric_limits<double>::epsilon()) {
					rank++;
					method_idx = method_rankings[rank].second;
					rank_count[crt_rank][method_idx]++;
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
	parallel_sort(top_3_rank_cnt, top_rank_cnt, cost_wo_cusp, cusp_cnt, cost_wo_worst_3, cost_wo_worst, method_cost, method_name, rank_count[0], rank_count[1], rank_count[2], rank_count[3], rank_count[4],
		rank_count[5], rank_count[6], rank_count[7], rank_count[8], rank_count[9],
		rank_count[10], rank_count[11], rank_count[12], rank_count[13], rank_count[14],
		rank_count[15]);
	// Save results
	write_csv("model_" + std::to_string(dataset_size) + "_wo_" + del_feat + "_on_" + std::to_string(point_num) + ".csv",
		method_name, top_3_rank_cnt, top_rank_cnt, cost_wo_cusp, cusp_cnt, cost_wo_worst_3, cost_wo_worst,
		method_cost, rank_count[0], rank_count[1], rank_count[2], rank_count[3], rank_count[4],
		rank_count[5], rank_count[6], rank_count[7], rank_count[8], rank_count[9],
		rank_count[10], rank_count[11], rank_count[12], rank_count[13], rank_count[14],
		rank_count[15]);
	std::cout << "total count: " << total_count;
	return 0;
}