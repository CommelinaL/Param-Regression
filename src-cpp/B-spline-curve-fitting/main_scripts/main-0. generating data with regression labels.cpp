#include <Eigen/Dense>
//#include <iostream>
#include <cmath>
#include <glut.h>
#include <gl/GLU.h>
//#include <glm/glm.hpp>
//#include<vector>
#include "tinynurbs.h"
//#include <fstream>
//#include <string>
#include <cmath>
#include <time.h>
#include <stdlib.h>
#include "omp.h"
#include "alg_Approximation.h"
#include "geo_Approximation.h"
#include "alg_Interpolation.h"
#include "CalculateCurve.h"

#include "ExtractFeatures.h"
#include "GenerateData.h"
#include "vec_file.hpp"

using namespace Eigen;
using namespace std;
#define POINTS_NUM 10
#define COTROL_NUM  4
#define KNOTS_NUM 8
#define TYPE_NUM 8
#define PI 3.1415926

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


int main(int argc, char** argv)
{
	vector<param_func> method_func = { UniformInterp, ChordInterp, CentripetalInterp, UniversalInterp,
	CorrectChordInterp,	RefinedCentripetalInterp, ModifiedChordInterp, ZCMInterp };

	int	num = 250000; // Quantity of raw data generated per round
	string project_root = PROJECT_ROOT;
	string DataFileName = project_root + "\\variable_length\\PointData_" + to_string(POINTS_NUM); // Raw data folder name
	string	SaveDataFileName = project_root + "\\variable_length\\Label_" + to_string(POINTS_NUM); // Generated labeled data folder name (final)
	create_directory_if_not_exists(DataFileName);
	create_directory_if_not_exists(SaveDataFileName);

	const clock_t begin_time = clock();


	srand((unsigned int)time(NULL));

	std::cout << "\n********************\n" << std::endl;
	GenerateDataDirect(num, POINTS_NUM, 6, DataFileName);

	omp_set_num_threads(20);
#pragma omp parallel for
	for (int k = 0; k < num; k++)
	{
		vector<glm::dvec3> pts; //为了输入和计算方便
		vector<vector<glm::dvec3>> CtrlPoints; // 控制点  对应关系： uniform chord centripetal universal correctchord RefinedCentripttal
		vector<vector<double>>  Knots; //节点
		vector<vector<double>> Param; //参数
		vector<tinynurbs::Curve<double>> crvs; //样条曲线
		vector<vector<glm::dvec3>> CurveTuringPoints;//存放曲线中的拐点
		vector<bool> LineTuringFlag;  //存放折线中的拐点标记


		/*计算对应文件名的编号*/
		std::string name;
		name = NumToStr(k);

		//name = NumToStr(4) + NumToStr(k);


		/*从文件中读取对应的数据*/
		fstream f;
		f.open(DataFileName + "/" + name + ".txt", ios::in);

		if (!f.is_open()) {
			cout << "file cann't be open." << endl;
			exit(0);
		}
		int fitting_num;
		f >> fitting_num;
		for (int i = 0; i < fitting_num; i++)
		{
			double x, y;
			f >> x >> y;
			pts.push_back(glm::dvec3(x, y, 0.0));
		}
		f.close();

		/*各种参数化方法的插值操作*/
		std::vector<std::pair<double, int>> method_rankings(9);
		double cost_best_heuristic = INFINITY;
		int best_idx = -1;
		std::vector<double> best_param;
		for (int i = 0; i < 8; i++) {
			std::vector<double> param;
			tinynurbs::Curve<double> crv;
			double cost_i = eval_param_func(method_func[i], pts, param, crv);
			if (cost_i < cost_best_heuristic) {
				cost_best_heuristic = cost_i;
				best_idx = i;
				best_param.clear();
				best_param.assign(param.begin(), param.end());
			}
		}
		std::string item_dir = SaveDataFileName + "/" + name;
		writeVectorToFile(best_param, item_dir + "/best.bin");
		copyFile(DataFileName + "/" + name + ".txt", item_dir + "/point_data.txt");

	}

	float seconds = float(clock() - begin_time) / 1000;
	std::cout << seconds << std::endl;


	return(0);

}
