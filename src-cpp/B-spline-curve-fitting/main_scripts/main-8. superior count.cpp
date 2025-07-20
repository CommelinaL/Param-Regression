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
	int point_num = 30;
	int dataset_size = 250000;
	std::string model_name = "MLP";
	std::string model_file_name = "MLP with manual feature.bin";
	std::string del_feat = "npc";
	std::string DataDir = "D:\\BSplineLearning\\sequential_data\\test_" + std::to_string(point_num) + "\\"; // The directory where sample is located
	std::string PredDir = "D:\\BSplineLearning\\pseudo_label\\seq_pred\\data_" + std::to_string(dataset_size) + "_wo_" + del_feat + "\\test_" + std::to_string(point_num) + "\\";
	std::string OtherCostDir = "D:\\BSplineLearning\\pseudo_label\\cost\\test_" + std::to_string(point_num) + "\\";
	std::string CostDir = "D:\\BSplineLearning\\pseudo_label\\cost\\data_" + std::to_string(dataset_size) + "_wo_" + del_feat + "\\" + model_name + "\\test_" + std::to_string(point_num) + "\\";
	create_directory_if_not_exists(CostDir);
	int superior_cnt = 0;
	std::string data_dir_tpl = DataDir + "*";
	int total_count = 0;
	double max_delta = -INFINITY;
	int max_delta_id = -1;
	std::vector<param_func> method_func = { UniformInterp, ChordInterp, CentripetalInterp, UniversalInterp,
	CorrectChordInterp,	RefinedCentripetalInterp, ModifiedChordInterp, ZCMInterp };
	WIN32_FIND_DATAA findFileData;
	HANDLE hFind = FindFirstFileA(data_dir_tpl.c_str(), &findFileData);
	if (hFind == INVALID_HANDLE_VALUE) {
		std::cerr << "No files found according to the template: " << data_dir_tpl << std::endl;
		return -1;
	}
	do {
		std::string PointPath = DataDir + findFileData.cFileName;
		std::string file_id = findFileData.cFileName;
		if (std::string(findFileData.cFileName) == "." || std::string(findFileData.cFileName) == "..") continue;
		file_id.erase(file_id.size() - 4);
		int id;
		try {
			id = std::stoi(file_id);
		}
		catch (const std::invalid_argument& e) {
			std::cerr << "Invalid argument: " << e.what() << " " << file_id << std::endl;
			continue;
		}
		total_count++;
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
			std::string cost_regress_path = CostDir + "Regressor" + "\\" + file_id + ".bin";
			std::string curvature_regress_path = CostDir + "Regressor" + "\\" + file_id + "_crv.bin";
			double cost_regress = 0, curvature_regress = 0;
			std::string pred_path = PredDir + file_id + "\\" + model_file_name;
			std::vector<double> regress_intervals;
			readVectorFromFile(regress_intervals, pred_path);
			std::ifstream in_crv_regress_file(curvature_regress_path, std::ios::binary);
			if (in_crv_regress_file.is_open()) {
				in_crv_regress_file.read(reinterpret_cast<char*>(&curvature_regress), sizeof(double));
				in_crv_regress_file.close();
				std::ifstream in_cost_regress_file(cost_regress_path, std::ios::binary);
				in_cost_regress_file.read(reinterpret_cast<char*>(&cost_regress), sizeof(double));
				in_cost_regress_file.close();
			}
			else {
				param_func regress_func = [&](const std::vector<glm::dvec3>& pts, int degree,
					std::vector<double>& knots, std::vector<double>& param)
					{ return RegressorInterp(pts, degree, knots, param, regress_intervals); };
				std::vector<double> param_regress;
				tinynurbs::Curve<double> crv_regress;
				cost_regress = eval_param_func(regress_func, pts, param_regress, crv_regress);
				curvature_regress = eval_curvature(pts, param_regress, crv_regress);
				std::ofstream out_cost_regress_file(cost_regress_path, std::ios::binary);
				create_directory_if_not_exists(cost_regress_path, true);
				out_cost_regress_file.write(reinterpret_cast<const char*>(&cost_regress), sizeof(double));
				out_cost_regress_file.close();
				std::ofstream out_crv_regress_file(curvature_regress_path, std::ios::binary);
				out_crv_regress_file.write(reinterpret_cast<const char*>(&curvature_regress), sizeof(double));
				out_crv_regress_file.close();
			}
			assert(cost_regress > std::numeric_limits<double>::epsilon());
			// Testing local loss of regressors and heuristic methods
			std::vector<double> best_heuristic_intervals;
			std::string best_heuristic_path = "D:\\BSplineLearning\\pseudo_label\\seq_pred\\test_" + std::to_string(point_num) + "\\" + file_id + "\\Label.bin";
			readVectorFromFile(best_heuristic_intervals, best_heuristic_path);
			if (best_heuristic_intervals.empty()) {
				for (int j = 0; j < point_num - 4 + 1; j++) {
					double cost_best_heuristic = INFINITY;
					int best_idx = -1;
					std::vector<double> best_param;
					for (int i = 0; i < 8; i++) {
						std::vector<double> param;
						tinynurbs::Curve<double> crv;
						std::vector<glm::dvec3> tmp_pts;
						tmp_pts.clear();
						tmp_pts.assign(pts.begin() + j, pts.begin() + j + 4);
						double cost_local = eval_param_func(method_func[i], tmp_pts, param, crv);
						if ((cost_local < cost_best_heuristic) && (param[1] == param[1]) && (param[2] == param[2])) {
							cost_best_heuristic = cost_local;
							best_idx = i;
							best_param.clear();
							best_param.assign(param.begin(), param.end());
						}
					}
					best_heuristic_intervals.push_back(best_param[1] - best_param[0]);
					best_heuristic_intervals.push_back(best_param[2] - best_param[1]);
					best_heuristic_intervals.push_back(best_param[3] - best_param[2]);
				}
				writeVectorToFile(best_heuristic_intervals, best_heuristic_path);
			}
			double cost_best_heuristic = 0, curvature_best_heuristic = 0;
			std::string cost_best_heuristic_path = OtherCostDir + "Label" + "\\" + file_id + ".bin";
			std::string curvature_best_heuristic_path = OtherCostDir + "Label" + "\\" + file_id + "_crv.bin";
			std::ifstream in_crv_best_heuristic_file(curvature_best_heuristic_path, std::ios::binary);
			if (in_crv_best_heuristic_file.is_open()) {
				in_crv_best_heuristic_file.read(reinterpret_cast<char*>(&curvature_best_heuristic), sizeof(double));
				in_crv_best_heuristic_file.close();
				std::ifstream in_cost_best_heuristic_file(cost_best_heuristic_path, std::ios::binary);
				in_cost_best_heuristic_file.read(reinterpret_cast<char*>(&cost_best_heuristic), sizeof(double));
				in_cost_best_heuristic_file.close();
			}
			else {
				param_func best_heuristic_func = [&](const std::vector<glm::dvec3>& pts, int degree,
					std::vector<double>& knots, std::vector<double>& param)
					{ return RegressorInterp(pts, degree, knots, param, best_heuristic_intervals); };
				std::vector<double> param;
				tinynurbs::Curve<double> crv;
				cost_best_heuristic = eval_param_func(best_heuristic_func, pts, param, crv);
				curvature_best_heuristic = eval_curvature(pts, param, crv);
				std::ofstream out_best_heuristic_file(cost_best_heuristic_path, std::ios::binary);
				create_directory_if_not_exists(cost_best_heuristic_path, true);
				out_best_heuristic_file.write(reinterpret_cast<const char*>(&cost_best_heuristic), sizeof(double));
				out_best_heuristic_file.close();
				std::ofstream crv_out_best_heuristic_file(curvature_best_heuristic_path, std::ios::binary);
				crv_out_best_heuristic_file.write(reinterpret_cast<const char*>(&curvature_best_heuristic), sizeof(double));
				crv_out_best_heuristic_file.close();
			}
			assert(cost_best_heuristic > std::numeric_limits<double>::epsilon());
			// Update the counts based on the sorted rankings
			if (cost_regress < cost_best_heuristic) {
				superior_cnt++;
			}
		}
		else std::cerr << "Failed to open file: " << PointPath << std::endl;
	} while (FindNextFileA(hFind, &findFileData) != 0);
	FindClose(hFind);
	// Save results
	std::ofstream best_sample_file("superior_cnt_" + std::to_string(point_num) + "_" + model_name + "_wo_" + del_feat + "_" + std::to_string(dataset_size) + ".txt");
	best_sample_file << "Superior count: " << superior_cnt << std::endl;
	best_sample_file.close();
	return 0;
}
