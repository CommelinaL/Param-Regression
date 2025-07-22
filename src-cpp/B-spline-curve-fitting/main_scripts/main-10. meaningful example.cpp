#include <Eigen/Dense>
#include <cmath>
#include <Windows.h>
#include "tinynurbs.h"
#include <cmath>
#include <stdlib.h>
#include <filesystem>
#include "omp.h"
#include "alg_Approximation.h"
#include "geo_Approximation.h"

#include "ExtractFeatures.h"
#include "GenerateData.h"
#include "vec_file.hpp"
#include <mutex>
#define OPENGA_EXTERN_LOCAL_VARS
std::mutex mtx_rand;
#include "param_knot_GA.hpp"

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

void writeCrv(const tinynurbs::Curve<double>& crv, const std::string output_dir) {
	create_directory_if_not_exists(output_dir);
	std::string ctrl_pts_path = output_dir + "\\control_points.txt";
	std::ofstream ctrl_pts_file(ctrl_pts_path);
	for (int i = 0; i < crv.control_points.size(); i++) {
		ctrl_pts_file << crv.control_points[i].x << " " << crv.control_points[i].y << std::endl;
	}
	ctrl_pts_file.close();
	std::string knots_path = output_dir + "\\knots.txt";
	std::ofstream knots_file(knots_path);
	for (int i = 0; i < crv.knots.size(); i++) {
		knots_file << crv.knots[i] << std::endl;
	}
	knots_file.close();
}

void ClassifierInterval(const std::vector<glm::dvec3>& points, const std::vector<int>& minis,
	std::vector<double>& intervals) {
	intervals.clear();
	for (int i = 0; i < points.size() - 3; i++) {
		std::vector<double> Knots;
		std::vector<double> Param;
		std::vector<glm::dvec3> tmpPoints = { points[i], points[i + 1], points[i + 2], points[i + 3] };
		if (minis[i] == 0) {
			UniversalInterp(tmpPoints, 3, Knots, Param);
		}
		else if (minis[i] == 1) {
			CorrectChordInterp(tmpPoints, 3, Knots, Param);
		}
		else {
			RefinedCentripetalInterp(tmpPoints, 3, Knots, Param);
		}
		intervals.push_back(Param[1] - Param[0]);
		intervals.push_back(Param[2] - Param[1]);
		intervals.push_back(Param[3] - Param[2]);
	}
}

void LocalCurves(const std::vector<glm::dvec3>& points,
	const std::vector<double>& intervals,
	std::vector<tinynurbs::Curve<double>>& crvs,
	std::vector<double>& local_costs) {
	crvs.clear();
	local_costs.clear();
	for (int i = 0; i < points.size() - 3; i++) {
		tinynurbs::Curve<double> tmp_crv;
		std::vector<double> tmp_param = { 0, intervals[i * 3], intervals[i * 3] + intervals[i * 3 + 1],
			intervals[i * 3] + intervals[i * 3 + 1] + intervals[i * 3 + 2] };
		std::vector<glm::dvec3> tmp_pts = { points[i], points[i + 1], points[i + 2], points[i + 3] };
		local_costs.push_back(eval_param(tmp_pts, tmp_param, tmp_crv));
		crvs.push_back(tmp_crv);
	}
}

int main(int argc, char** argv)
{
	std::string file_id = "clash";
	std::string del_feat = "npc";
	std::string pred_path = "D:\\BSplineLearning\\Param-Regression\\meaningful_examples\\regress\\" + file_id + ".bin";
	std::string class_path = "D:\\BSplineLearning\\Param-Regression\\meaningful_examples\\class\\" + file_id + "-r.txt";
	std::string output_dir = "D:\\BSplineLearning\\Param-Regression\\crv\\meaningful_examples\\" + file_id + "\\";
	std::string local_output_dir = output_dir + "local\\";
	std::vector<double> method_cost(12, 0);
	std::vector<std::vector<double>> params(12);
	std::vector<tinynurbs::Curve<double>> method_crv(12);
	std::vector<std::vector<tinynurbs::Curve<double>>> local_crv(4);
	std::vector<std::vector<double>> local_cost(4);
	std::vector<std::vector<double>> intervals(4);
	std::vector<std::string> method_name = { "uniform", "chord", "centripetal", "universal",
	"foley", "fang", "modified_chord", "zcm", // Heuristic methods
	"Classifier", "Regressor", "Label", "Classify_best" };
	std::vector<std::string> local_name = { "Classifier", "Regressor", "Label" };
	std::vector<param_func> method_func = { UniformInterp, ChordInterp, CentripetalInterp, UniversalInterp,
	CorrectChordInterp,	RefinedCentripetalInterp, ModifiedChordInterp, ZCMInterp };
	std::string PointPath = "D:\\BSplineLearning\\Param-Regression\\meaningful_examples\\" + file_id + ".txt";
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
		for (int i = 0; i < 8; i++) {
			method_cost[i] = eval_param_func(method_func[i], pts, params[i], method_crv[i]);
			assert(method_cost[i] > std::numeric_limits<double>::epsilon());
		}
		// Parametrization with classifier
		std::ifstream class_file(class_path);
		vector<int> minis;
		minis.resize(pts.size() - 3);
		for (int i = 0; i < pts.size() - 3; i++)
		{
			class_file >> minis[i];
			cout << minis[i] << " ";
		}
		cout << endl;
		param_func classifier_func = [&](const std::vector<glm::dvec3>& pts, int degree,
			std::vector<double>& knots, std::vector<double>& param)
			{ return ClassfierInterp(pts, degree, knots, param, minis); };
		method_cost[8] = eval_param_func(classifier_func, pts, params[8], method_crv[8]);
		assert(method_cost[8] > std::numeric_limits<double>::epsilon());
		// Classifier local
		ClassifierInterval(pts, minis, intervals[0]);
		LocalCurves(pts, intervals[0], local_crv[0], local_cost[0]);
		// Testing regressor
		readVectorFromFile(intervals[1], pred_path);
		param_func regress_func = [&](const std::vector<glm::dvec3>& pts, int degree,
			std::vector<double>& knots, std::vector<double>& param)
			{ return RegressorInterp(pts, degree, knots, param, intervals[1]); };
		method_cost[9] = eval_param_func(regress_func, pts, params[9], method_crv[9]);
		assert(method_cost[9] > std::numeric_limits<double>::epsilon());
		// Regressor local
		LocalCurves(pts, intervals[1], local_crv[1], local_cost[1]);
		// Interpolating with best local heuristic methods
		std::string best_heuristic_path = "D:\\BSplineLearning\\pseudo_label\\seq_pred\\meaningful_examples\\test_" + std::to_string(point_num) + "\\" + file_id + "\\Label.bin";
		readVectorFromFile(intervals[2], best_heuristic_path);
		if (intervals[2].empty()) {
			for (int j = 0; j < point_num - 3; j++) {
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
				intervals[2].push_back(best_param[1] - best_param[0]);
				intervals[2].push_back(best_param[2] - best_param[1]);
				intervals[2].push_back(best_param[3] - best_param[2]);
			}
			writeVectorToFile(intervals[2], best_heuristic_path);
		}
		param_func best_heuristic_func = [&](const std::vector<glm::dvec3>& pts, int degree,
			std::vector<double>& knots, std::vector<double>& param)
			{ return RegressorInterp(pts, degree, knots, param, intervals[2]); };
		method_cost[10] = eval_param_func(best_heuristic_func, pts, params[10], method_crv[10]);
		assert(method_cost[10] > std::numeric_limits<double>::epsilon());
		LocalCurves(pts, intervals[2], local_crv[2], local_cost[2]);
		// Classification labels
		for (int j = 0; j < point_num - 3; j++) {
			double cost_best_heuristic = INFINITY;
			int best_idx = -1;
			std::vector<double> best_param;
			for (int i = 3; i < 6; i++) {
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
			intervals[3].push_back(best_param[1] - best_param[0]);
			intervals[3].push_back(best_param[2] - best_param[1]);
			intervals[3].push_back(best_param[3] - best_param[2]);
		}
		param_func best_class_func = [&](const std::vector<glm::dvec3>& pts, int degree,
			std::vector<double>& knots, std::vector<double>& param)
			{ return RegressorInterp(pts, degree, knots, param, intervals[3]); };
		method_cost[11] = eval_param_func(best_class_func, pts, params[11], method_crv[11]);
		assert(method_cost[11] > std::numeric_limits<double>::epsilon());
		LocalCurves(pts, intervals[3], local_crv[3], local_cost[3]);
		for (int i = 0; i < method_crv.size(); i++) {
			std::string crv_dir = output_dir + std::to_string(i);
			writeCrv(method_crv[i], crv_dir);
			std::string param_path = crv_dir + "\\param.txt";
			std::ofstream param_file(param_path);
			for (int j = 0; j < params[i].size(); j++) {
				param_file << params[i][j] << std::endl;
			}
			param_file.close();
		}
		std::string metric_path = output_dir + "metric_res.txt";
		std::ofstream metric_file(metric_path);
		for (int i = 0; i < method_cost.size(); i++) {
			metric_file << method_cost[i] << std::endl;
		}
		metric_file.close();
		std::string data_pts_path = output_dir + "data_points.txt";
		std::ofstream data_pts_file(data_pts_path);
		for (int i = 0; i < pts.size(); i++) {
			data_pts_file << pts[i].x << " " << pts[i].y << std::endl;
		}
		data_pts_file.close();
		// Save local interpolation curves
		for (int i = 0; i < local_crv.size(); i++) {
			std::string method_local_dir = local_output_dir + std::to_string(i) + "\\";
			std::string local_metric_path = method_local_dir + "metric_res.txt";
			std::ofstream local_metric_file(local_metric_path);
			cout << i << endl;
			for (int j = 0; j < local_cost[i].size(); j++) {
				cout << local_cost[i][j] << " ";
				local_metric_file << local_cost[i][j] << std::endl;
			}
			cout << endl;
			local_metric_file.close();
			for (int j = 0; j < local_crv[i].size(); j++) {
				std::string j_dir = method_local_dir + std::to_string(j) + "\\";
				writeCrv(local_crv[i][j], j_dir);
				std::string param_path = j_dir + "\\param.txt";
				std::ofstream param_file(param_path);
				double p_local = 0;
				for (int k = 0; k < 3; k++) {
					param_file << p_local << std::endl;
					p_local += intervals[i][j * 3 + k];
				}
				param_file << p_local << std::endl;
				param_file.close();
			}
		}
	}
	else std::cerr << "Failed to open file: " << PointPath << std::endl;
}