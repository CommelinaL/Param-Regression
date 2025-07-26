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

void get_knot(param_func& func, const std::vector<glm::dvec3>& points,
	std::vector<double>& param, tinynurbs::Curve<double>& crv, const std::string path_output,
    const std::string name) {
	// Initialization
	crv.knots.clear();
	param.clear();
	crv.control_points.clear();
	crv.degree = 3;
	func(points, 3, crv.knots, param);
	// calculate knots
	run_knot_test(true, true, 1000, points, name, path_output, 100, param, 1e-4);
	readVectorFromFile(crv.knots, path_output);
}

double eval_param(const std::vector<double>& param, const std::vector<glm::dvec3>& points,
	tinynurbs::Curve<double>& crv) {
	// Initialization
	crv.control_points.clear();
	crv.degree = 3;
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
	std::string project_root = PROJECT_ROOT;
	std::string DataDir = project_root + "\\sequential_data\\test_" + std::to_string(point_num) + "\\"; // The directory where sample is located
	std::string PredDir = project_root + "\\pseudo_label\\seq_pred\\data_" + std::to_string(dataset_size) + "_wo_" + del_feat + "\\test_" + std::to_string(point_num) + "\\";
	std::string OtherCostDir = project_root + "\\pseudo_label\\cost\\ga\\test_" + std::to_string(point_num) + "\\";
	std::string CostDir = project_root + "\\pseudo_label\\cost\\ga\\data_" + std::to_string(dataset_size) + "_wo_" + del_feat + "\\" + model_name + "\\test_" + std::to_string(point_num) + "\\";
	std::string OtherKnotDir = project_root + "\\pseudo_label\\ga_knots\\test_" + std::to_string(point_num) + "\\";
	std::string KnotDir = project_root + "\\pseudo_label\\ga_knots\\data_" + std::to_string(dataset_size) + "_wo_" + del_feat + "\\" + model_name + "\\test_" + std::to_string(point_num) + "\\";
	create_directory_if_not_exists(CostDir);
	double local_avg_wo_outliers = 0;
	int local_outlier_cnt = 0;
	std::string data_dir_tpl = DataDir + "*";
	std::vector<double> method_cost(13, 0);
	std::vector<double> cost_wo_cusp(13, 0);
	std::vector<double> cost_cusp(13, 0);
	std::vector<int> cusp_cnt(13, 0);
	std::vector<double> worst_cost(13, 0);
	std::vector<double> sec_worst(13, 0);
	std::vector<double> thr_worst(13, 0);
	std::vector<double> cost_wo_worst(13, 0);
	std::vector<double> cost_wo_worst_3(13, 0);
	std::vector<int> class_failure;
	std::vector<int> regress_failure;
	std::vector<int> label_failure;
	std::vector<int> class_cusp;
	std::vector<int> regress_cusp;
	std::vector<int> label_cusp;
	std::vector<std::vector<int>> rank_count;
	rank_count.clear();
	for (int i = 0; i < 13; i++) {
		std::vector<int> tmp(13, 0);
		rank_count.push_back(tmp);
	}
	int total_count = 0;
	double max_delta = -INFINITY;
	int max_delta_id = -1;
	std::vector<std::string> method_name = { "uniform", "chord", "centripetal", "universal",
	"foley", "fang", "modified_chord", "zcm", // Heuristic methods
	"Classifier", "Regressor", "Label","Regressor local",  "Label_local" };
	std::cout << method_name.size() << std::endl;
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
		if (total_count >= 100) {
			break;
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
			// Parametrization with heuristic methods
			std::vector<std::pair<double, int>> method_rankings(13);
			for (int i = 0; i < 8; i++) {
				std::string curvature_path = OtherCostDir + method_name[i] + "\\" + file_id + "_crv.bin";
				std::string cost_path = OtherCostDir + method_name[i] + "\\" + file_id + ".bin";
				double cost_i = 0, curvature_i = 0;
				std::ifstream in_crv_file(curvature_path, std::ios::binary);
				if (in_crv_file.is_open()) {
					in_crv_file.read(reinterpret_cast<char*>(&curvature_i), sizeof(double));
					std::ifstream in_cost_file(cost_path, std::ios::binary);
					in_cost_file.read(reinterpret_cast<char*>(&cost_i), sizeof(double));
					in_cost_file.close();
				}
				else {
					std::vector<double> param;
					tinynurbs::Curve<double> crv;
					std::string knot_path = OtherKnotDir + method_name[i] + "\\" + file_id + ".bin";
					(method_func[i])(pts, 3, crv.knots, param);
					crv.knots.clear();
					readVectorFromFile(crv.knots, knot_path);
					if (crv.knots.empty()) {
						get_knot(method_func[i], pts, param, crv, knot_path, method_name[i]);
					}
					cost_i = eval_param(param, pts, crv);
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
			// Parametrization with classifier
			std::string cost_cls_path = OtherCostDir + method_name[8] + "\\" + file_id + ".bin";
			std::string curvature_cls_path = OtherCostDir + method_name[8] + "\\" + file_id + "_crv.bin";
			double cost_class = 0, curvature_class = 0;
			std::ifstream in_crv_cls_file(curvature_cls_path, std::ios::binary);
			if (in_crv_cls_file.is_open()) {
				in_crv_cls_file.read(reinterpret_cast<char*>(&curvature_class), sizeof(double));
				in_crv_cls_file.close();
				std::ifstream in_cost_cls_file(cost_cls_path, std::ios::binary);
				in_cost_cls_file.read(reinterpret_cast<char*>(&cost_class), sizeof(double));
				in_cost_cls_file.close();
			}
			else {
				std::string class_path = project_root + "\\pseudo_label\\seq_pred\\class\\test_" + std::to_string(point_num) + "\\" + file_id + "-r.txt";
				std::ifstream class_file(class_path);
				vector<int> minis;
				minis.resize(pts.size() - 3);
				for (int i = 0; i < pts.size() - 3; i++)
				{
					class_file >> minis[i];
				}
				param_func classifier_func = [&](const std::vector<glm::dvec3>& pts, int degree,
					std::vector<double>& knots, std::vector<double>& param)
					{ return ClassfierInterp(pts, degree, knots, param, minis); };
				std::vector<double> param_class;
				tinynurbs::Curve<double> crv_class;
				classifier_func(pts, 3, crv_class.knots, param_class);
				crv_class.knots.clear();
				std::string knot_class_path = OtherKnotDir + method_name[8] + "\\" + file_id + ".bin";
				readVectorFromFile(crv_class.knots, knot_class_path);
				if (crv_class.knots.empty()) {
					get_knot(classifier_func, pts, param_class, crv_class, knot_class_path, method_name[8]);
				}
				cost_class = eval_param(param_class, pts, crv_class);
				curvature_class = eval_curvature(pts, param_class, crv_class);
				create_directory_if_not_exists(cost_cls_path, true);
				std::ofstream out_cost_cls_file(cost_cls_path, std::ios::binary);
				out_cost_cls_file.write(reinterpret_cast<const char*>(&cost_class), sizeof(double));
				out_cost_cls_file.close();
				std::ofstream out_curvature_cls_file(curvature_cls_path, std::ios::binary);
				out_curvature_cls_file.write(reinterpret_cast<const char*>(&curvature_class), sizeof(double));
				out_curvature_cls_file.close();
			}
			assert(cost_class > std::numeric_limits<double>::epsilon());
			method_cost[8] += cost_class;
			method_rankings[8] = { cost_class, 8 };
			if (curvature_class > 1e4) {
				cusp_cnt[8]++;
				cost_cusp[8] += cost_class;
				class_cusp.push_back(id);
			}
			if (cost_class > worst_cost[8]) {
				thr_worst[8] = sec_worst[8];
				sec_worst[8] = worst_cost[8];
				worst_cost[8] = cost_class;
			}
			else if (cost_class > sec_worst[8]) {
				thr_worst[8] = sec_worst[8];
				sec_worst[8] = cost_class;
			}
			else if (cost_class > thr_worst[8]) {
				thr_worst[8] = cost_class;
			}
			// Testing regressors
			std::string cost_regress_path = CostDir + method_name[9] + "\\" + file_id + ".bin";
			std::string curvature_regress_path = CostDir + method_name[9] + "\\" + file_id + "_crv.bin";
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
				regress_func(pts, 3, crv_regress.knots, param_regress);
				crv_regress.knots.clear();
				std::string knot_regress_path = KnotDir + method_name[9] + "\\" + file_id + ".bin";
				readVectorFromFile(crv_regress.knots, knot_regress_path);
				if (crv_regress.knots.empty()) {
					get_knot(regress_func, pts, param_regress, crv_regress, knot_regress_path, method_name[9]);
				}
				cost_regress = eval_param(param_regress, pts, crv_regress);
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
			method_cost[9] += cost_regress;
			method_rankings[9] = { cost_regress, 9 };
			if (curvature_regress > 1e4) {
				cusp_cnt[9]++;
				cost_cusp[9] += cost_regress;
				regress_cusp.push_back(id);
			}
			if (cost_regress > worst_cost[9]) {
				thr_worst[9] = sec_worst[9];
				sec_worst[9] = worst_cost[9];
				worst_cost[9] = cost_regress;
			}
			else if (cost_regress > sec_worst[9]) {
				thr_worst[9] = sec_worst[9];
				sec_worst[9] = cost_regress;
			}
			else if (cost_regress > thr_worst[9]) {
				thr_worst[9] = cost_regress;
			}
			//// Testing local loss of regressors and heuristic methods
			//std::vector<double> best_heuristic_intervals;
			//std::string best_heuristic_path = project_root + "\\pseudo_label\\seq_pred\\test_" + std::to_string(point_num) + "\\" + file_id + "\\Label.bin";
			//readVectorFromFile(best_heuristic_intervals, best_heuristic_path);
			//if (best_heuristic_intervals.empty()) {
			//	for (int j = 0; j < point_num - 4 + 1; j++) {
			//		double cost_best_heuristic = INFINITY;
			//		int best_idx = -1;
			//		std::vector<double> best_param;
			//		for (int i = 0; i < 8; i++) {
			//			std::vector<double> param;
			//			tinynurbs::Curve<double> crv;
			//			std::vector<glm::dvec3> tmp_pts;
			//			tmp_pts.clear();
			//			tmp_pts.assign(pts.begin() + j, pts.begin() + j + 4);
			//			double cost_local = eval_param_func(method_func[i], tmp_pts, param, crv);
			//			if ((cost_local < cost_best_heuristic) && (param[1] == param[1]) && (param[2] == param[2])) {
			//				cost_best_heuristic = cost_local;
			//				best_idx = i;
			//				best_param.clear();
			//				best_param.assign(param.begin(), param.end());
			//			}
			//		}
			//		best_heuristic_intervals.push_back(best_param[1] - best_param[0]);
			//		best_heuristic_intervals.push_back(best_param[2] - best_param[1]);
			//		best_heuristic_intervals.push_back(best_param[3] - best_param[2]);
			//	}
			//	writeVectorToFile(best_heuristic_intervals, best_heuristic_path);
			//}
			//double cost_local_regress = 0, cost_local_heuristic = 0;
			//double local_outlier_sum = 0;
			//int local_cnt_j = 0;
			//for (int j = 0; j < point_num - 3; j++) {
			//	// best local loss of heuristic methods
			//	std::string cost_local_heuristic_path = OtherCostDir + "best_heuristic\\" + file_id + "_" + std::to_string(j) + ".bin";
			//	std::ifstream in_cost_local_heuristic_file(cost_local_heuristic_path, std::ios::binary);
			//	double cost_local_heuristic_j = 0;
			//	if (in_cost_local_heuristic_file.is_open()) {
			//		in_cost_local_heuristic_file.read(reinterpret_cast<char*>(&cost_local_heuristic_j), sizeof(double));
			//		in_cost_local_heuristic_file.close();
			//	}
			//	else {
			//		MySolution gene;
			//		MyMiddleCost cost;
			//		gene.param.push_back(0);
			//		gene.param.push_back(best_heuristic_intervals[j * 3]);
			//		gene.param.push_back(best_heuristic_intervals[j * 3] + best_heuristic_intervals[j * 3 + 1]);
			//		gene.param.push_back(best_heuristic_intervals[j * 3] + best_heuristic_intervals[j * 3 + 1] + best_heuristic_intervals[j * 3 + 2]);
			//		std::vector<glm::dvec3> tmp_pts;
			//		tmp_pts.assign(pts.begin() + j, pts.begin() + j + 4);
			//		eval_solution_tpl(gene, cost, tmp_pts);
			//		cost_local_heuristic_j = cost.cost_avg;
			//		create_directory_if_not_exists(cost_local_heuristic_path, true);
			//		std::ofstream out_cost_local_heuristic_file(cost_local_heuristic_path, std::ios::binary);
			//		out_cost_local_heuristic_file.write(reinterpret_cast<const char*>(&cost_local_heuristic_j), sizeof(double));
			//		out_cost_local_heuristic_file.close();
			//	}
			//	cost_local_heuristic += cost_local_heuristic_j;
			//	// local loss of regressors
			//	readVectorFromFile(regress_intervals, pred_path);
			//	std::string cost_local_regress_path = CostDir + method_name[9] + "\\" + file_id + "_" + std::to_string(j) + ".bin";
			//	std::ifstream in_cost_local_regress_file(cost_local_regress_path, std::ios::binary);
			//	double cost_local_regress_j = 0;
			//	if (in_cost_local_regress_file.is_open()) {
			//		in_cost_local_regress_file.read(reinterpret_cast<char*>(&cost_local_regress_j), sizeof(double));
			//		in_cost_local_regress_file.close();
			//	}
			//	else {
			//		MySolution gene;
			//		MyMiddleCost cost;
			//		gene.param.push_back(0);
			//		gene.param.push_back(regress_intervals[j * 3]);
			//		gene.param.push_back(regress_intervals[j * 3] + regress_intervals[j * 3 + 1]);
			//		gene.param.push_back(regress_intervals[j * 3] + regress_intervals[j * 3 + 1] + regress_intervals[j * 3 + 2]);
			//		std::vector<glm::dvec3> tmp_pts;
			//		tmp_pts.assign(pts.begin() + j, pts.begin() + j + 4);
			//		eval_solution_tpl(gene, cost, tmp_pts);
			//		cost_local_regress_j = cost.cost_avg;
			//		create_directory_if_not_exists(cost_local_regress_path, true);
			//		std::ofstream out_cost_local_regress_file(cost_local_regress_path, std::ios::binary);
			//		out_cost_local_regress_file.write(reinterpret_cast<const char*>(&cost_local_regress_j), sizeof(double));
			//		out_cost_local_regress_file.close();
			//	}
			//	if (cost_local_regress_j / cost_local_heuristic_j > 5) {
			//		local_outlier_cnt++;
			//		local_cnt_j++;
			//		local_outlier_sum += cost_local_regress_j;
			//	}
			//	cost_local_regress += cost_local_regress_j;
			//}
			//local_avg_wo_outliers += (cost_local_regress - local_outlier_sum) / (double)(point_num - 3 - local_cnt_j);
			//cost_local_regress /= point_num - 3;
			//method_cost[11] += cost_local_regress;
			//method_rankings[11] = { cost_local_regress, 11 };
			//cost_local_heuristic /= point_num - 3;
			//method_cost[12] += cost_local_heuristic;
			//method_rankings[12] = { cost_local_heuristic, 12 };
			// Testing the merged result of best heuristic parameterizations
			/*double cost_best_heuristic = 0, curvature_best_heuristic = 0;
			std::string cost_best_heuristic_path = OtherCostDir + method_name[10] + "\\" + file_id + ".bin";
			std::string curvature_best_heuristic_path = OtherCostDir + method_name[10] + "\\" + file_id + "_crv.bin";
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
			method_cost[10] += cost_best_heuristic;
			method_rankings[10] = { cost_best_heuristic, 10 };
			if (curvature_best_heuristic > 1e4) {
				cusp_cnt[10]++;
				cost_cusp[10] += cost_best_heuristic;
				label_cusp.push_back(id);
			}
			if (cost_best_heuristic > worst_cost[10]) {
				thr_worst[10] = sec_worst[10];
				sec_worst[10] = worst_cost[10];
				worst_cost[10] = cost_best_heuristic;
			}
			else if (cost_best_heuristic > sec_worst[10]) {
				thr_worst[10] = sec_worst[10];
				sec_worst[10] = cost_best_heuristic;
			}
			else if (cost_best_heuristic > thr_worst[10]) {
				thr_worst[10] = cost_best_heuristic;
			}*/
			// Sort the methods based on their costs
			std::sort(method_rankings.begin(), method_rankings.begin() + 10);
			// When regressor gives the relatively best result
			double delta;
			if (method_rankings[0].second == 9) {
				delta = method_rankings[1].first - cost_regress;
			}
			else {
				delta = method_rankings[0].first - cost_regress;
			}
			if (delta > max_delta) {
				max_delta = delta;
				max_delta_id = id;
			}
			// Update the counts based on the sorted rankings
			for (int rank = 0; rank < 10; rank++) {
				int method_idx = method_rankings[rank].second;
				rank_count[rank][method_idx]++;
				int crt_rank = rank;
				if (crt_rank >= 9 && method_idx == 8) {
					class_failure.push_back(id);
				}
				if (crt_rank >= 9 && method_idx == 9) {
					regress_failure.push_back(id);
				}
				if (crt_rank >= 9 && method_idx == 10) {
					label_failure.push_back(id);
				}
				while (rank + 1 < 11 && method_rankings[rank + 1].first - method_rankings[crt_rank].first < numeric_limits<double>::epsilon()) {
					rank++;
					method_idx = method_rankings[rank].second;
					rank_count[crt_rank][method_idx]++;
					if (crt_rank >= 9 && method_idx == 9) {
						regress_failure.push_back(id);
					}
					if (crt_rank >= 9 && method_idx == 10) {
						label_failure.push_back(id);
					}
				}
			}
		}
		else std::cerr << "Failed to open file: " << PointPath << std::endl;
	} while (FindNextFileA(hFind, &findFileData) != 0);
	FindClose(hFind);
	// Output results
	for (int i = 0; i < method_cost.size(); i++) {
		cost_wo_cusp[i] = (method_cost[i] - cost_cusp[i]) / (double)(total_count - cusp_cnt[i]);
		cost_wo_worst_3[i] = (method_cost[i] - worst_cost[i] - sec_worst[i] - thr_worst[i]) / (double)(total_count - 3);
		cost_wo_worst[i] = (method_cost[i] - worst_cost[i]) / (double)(total_count - 1);
		method_cost[i] /= (double)total_count;
	}
	local_avg_wo_outliers /= (double)total_count;
	std::cout << "local outlier cnt:" << local_outlier_cnt << endl;
	std::cout << "local avg wo outliers:" << local_avg_wo_outliers << endl;
	parallel_sort(cost_wo_cusp, cusp_cnt, cost_wo_worst_3, cost_wo_worst, method_cost, method_name, rank_count[0], rank_count[1], rank_count[2], rank_count[3], rank_count[4],
		rank_count[5], rank_count[6], rank_count[7], rank_count[8], rank_count[9],
		rank_count[10], rank_count[11], rank_count[12]);
	// Save results
	write_csv("test_ga_" + std::to_string(point_num) + "_" + model_name + "_wo_" + del_feat + "_" + std::to_string(dataset_size) + ".csv",
		method_name, cost_wo_cusp, cusp_cnt, cost_wo_worst_3, cost_wo_worst,
		method_cost, rank_count[0], rank_count[1], rank_count[2], rank_count[3], rank_count[4],
		rank_count[5], rank_count[6], rank_count[7], rank_count[8], rank_count[9],
		rank_count[10], rank_count[11], rank_count[12]);
	std::cout << "total count: " << total_count << std::endl;
	std::cout << "max_delta:" << max_delta << ", max_delta_id:" << max_delta_id << std::endl;
	std::ofstream best_sample_file("best_sample_ga_test_" + std::to_string(point_num) + "_" + model_name + "_wo_" + del_feat + "_" + std::to_string(dataset_size) + ".txt");
	best_sample_file << "max_delta:" << max_delta << ", max_delta_id:" << max_delta_id << std::endl;
	best_sample_file << "class failure:" << endl;
	for (int i = 0; i < class_failure.size(); i++) {
		best_sample_file << class_failure[i] << endl;
	}
	best_sample_file << "regress failure:" << endl;
	for (int i = 0; i < regress_failure.size(); i++) {
		best_sample_file << regress_failure[i] << endl;
	}
	best_sample_file << "label failure:" << endl;
	/*for (int i = 0; i < label_failure.size(); i++) {
		best_sample_file << label_failure[i] << endl;
	}*/
	best_sample_file << "class cusp:" << endl;
	for (int i = 0; i < class_cusp.size(); i++) {
		best_sample_file << class_cusp[i] << endl;
	}
	best_sample_file << "regress cusp:" << endl;
	for (int i = 0; i < regress_cusp.size(); i++) {
		best_sample_file << regress_cusp[i] << endl;
	}
	best_sample_file << "label cusp:" << endl;
	for (int i = 0; i < label_cusp.size(); i++) {
		best_sample_file << label_cusp[i] << endl;
	}
	best_sample_file.close();
	return 0;
}
