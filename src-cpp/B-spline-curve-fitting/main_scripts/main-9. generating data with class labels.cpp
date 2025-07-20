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

using namespace Eigen;
using namespace std;
#define POINTS_NUM 4
#define COTROL_NUM  4
#define KNOTS_NUM 8
#define TYPE_NUM 6
#define PI 3.1415926

/*opengl绘制nurbs曲线输入和计算时用到的量*/
vector<bool> g_LineTuringFlag;  //折线上的拐点标记
vector<vector<glm::dvec3>> g_CurveTuringPoints;//曲线上的拐点坐标
vector<glm::dvec3> g_pts; //数据点
vector<vector<glm::dvec3>> g_CtrlPoints; // 控制点  对应关系： uniform chord centripetal universal correctchord RefinedCentripttal
vector<vector<double>>  g_Knots; //节点矢量
vector<vector<vector<glm::dvec3>>> b_CtrlPoints; //节点细化之后的贝塞尔曲线的控制点
vector<vector<glm::dvec3>> knots_pts; //节点矢量对应的点
vector<int>  b_nb; //每条nurbs节点细化后的贝塞尔曲线的个数
double Cri_W[TYPE_NUM];  //保存opengl对应nurbs的评判指标
int type_num[TYPE_NUM] = { 0 };//保存所有数据中每种方法被选到的次数
vector<vector<double>> b_hausdorrf; //保存节点细化后的贝塞尔计算的豪斯多夫距离
vector<vector<double>> p_hausdorrf; //利用离散点计算的豪斯多夫距离

/*opengl绘制nurbs曲线用到的量*/
GLUnurbsObj* theNurb;
GLfloat ctrlpoints[COTROL_NUM][3];
GLfloat knots[KNOTS_NUM];
GLfloat dataPoints[POINTS_NUM][3];//被拟合的数据点
GLfloat knotsPoints[TYPE_NUM][POINTS_NUM - 2][3]; // 节点矢量对应的点 - 测试节点细化和节点矢量的对应关系
char type_name[TYPE_NUM][20] = { "Uniform ", "Chord ", "Centripetal ", "Universal ", "CorrectChord ", "RefinedCentripttal" };
double max_length, max_width, min_length, min_width, gap_width, gap_length;  //用于窗口呈现
bool draw_flag[TYPE_NUM] = { true }; // 每种方法对应nurbs是否绘制开关
bool draw_points_flag = true; //数据点绘制开关
bool draw_knots_flag = true; //节点矢量绘制开关
bool draw_line_flag = true; //折线绘制开关
GLfloat testdata[TYPE_NUM][3]; //测试用opengl数据



int main(int argc, char** argv)
{
	int id;

	//int draw_id = 0;
	//int draw_mode = 1;

	int	num; //每轮原始数据生成的数量 // Quantity of raw data generated per round
	int	train_num; //每个标签的训练数据量 // Training data quantity per label
	int test_num; // 每个标签的测试数据量 // Test data quantity per label
	string	FeatureLableFileName; //特征和标签文件夹名称 // Feature and label folder name
	string	DataFileName; //原始数据文件夹名称 // Raw data folder name
	string	SaveDataFileName;// 最终打好标签的生成数据文件夹名称 // Generated labeled data folder name


	std::cin >> num >> train_num >> test_num >> FeatureLableFileName >> DataFileName >> SaveDataFileName;


	const clock_t begin_time = clock();
	b_hausdorrf.resize(TYPE_NUM);
	p_hausdorrf.resize(TYPE_NUM);

	vector<int> TrainDataNum;
	vector<int> TestDataNum;
	for (int i = 0; i < 6; i++)
	{
		TrainDataNum.push_back(0);
		TestDataNum.push_back(0);
	}


	srand((unsigned int)time(NULL));

	std::cout << "\n********************\n" << std::endl;
	GenerateDataDirect(num, POINTS_NUM, 6, DataFileName);

	while (1) {

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

			vector<double> FeatureLabelData; //特征和标签的数据
			vector<double> Matrix; //评判指标的值

			/*计算对应文件名的编号*/
			std::string name;
			name = NumToStr(k);

			//name = NumToStr(4) + NumToStr(k);


			/*从文件中读取对应的数据*/
			fstream f;
			f.open(DataFileName + "/" + name + ".txt", ios::in);

			//std::string s_name = "TrainData100-4";
			//f.open(s_name + "/traincrv" + "/" + name + ".txt", ios::in);

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
			CtrlPoints.clear();
			Knots.clear();
			Param.clear();
			CtrlPoints.resize(TYPE_NUM);
			Knots.resize(TYPE_NUM);
			Param.resize(TYPE_NUM);

			/*基于 Uniform 的插值  */
			UniformInterp(pts, 3, Knots[0], Param[0]);
			GlobalInterp(pts, 3, Knots[0], Param[0], CtrlPoints[0]);

			/*基于 Chord 的插值 */
			ChordInterp(pts, 3, Knots[1], Param[1]);
			GlobalInterp(pts, 3, Knots[1], Param[1], CtrlPoints[1]);

			/*基于 Centripetal 的插值 */
			CentripetalInterp(pts, 3, Knots[2], Param[2]);
			GlobalInterp(pts, 3, Knots[2], Param[2], CtrlPoints[2]);

			/*基于 Universal 的插值 */
			UniversalInterp(pts, 3, Knots[3], Param[3]);
			GlobalInterp(pts, 3, Knots[3], Param[3], CtrlPoints[3]);

			/*基于 CorrectChord 的插值 */
			CorrectChordInterp(pts, 3, Knots[4], Param[4]);
			GlobalInterp(pts, 3, Knots[4], Param[4], CtrlPoints[4]);

			/*基于 RefinedCentripetal 的插值*/
			RefinedCentripetalInterp(pts, 3, Knots[5], Param[5]);
			GlobalInterp(pts, 3, Knots[5], Param[5], CtrlPoints[5]);

			/*保存节点矢量和控制点到crvs方便后续计算*/
			crvs.clear();
			crvs.resize(TYPE_NUM);
			for (int i = 0; i < TYPE_NUM; i++)
			{
				crvs[i].degree = 3;
				for (int j = 0; j < KNOTS_NUM; j++)
				{
					crvs[i].knots.push_back(Knots[i][j]);
				}

				for (int j = 0; j < COTROL_NUM; j++)
				{
					crvs[i].control_points.push_back(glm::dvec3(CtrlPoints[i][j].x, CtrlPoints[i][j].y, 0));
				}

			}



			//评判标准的计算
			/*
			//计算曲率的平方沿曲线的积分
			std::vector<double> totalCrvIntegrals; //用分段贝塞尔数值积分计算结果
			std::vector<double> CrvIntegrals;  //直接利用数值积分计算的结果
			totalCrvIntegrals.resize(TYPE_NUM);
			for (int i = 0; i < TYPE_NUM; i++)
			{
				double CurvatureIntegral = calculateCurveCurvatureIntegral(crvs[i], 20);
				CrvIntegrals.push_back(CurvatureIntegral);
			}
			*/



			/*豪斯多夫距离的计算*/
			/*
			std::vector<std::vector<double>> Hausdorrf;
			Hausdorrf.resize(TYPE_NUM);

			std::vector<double> maxHausdorrf;
			std::vector<double> avgHausdorrf;

			maxHausdorrf.resize(TYPE_NUM);
			avgHausdorrf.resize(TYPE_NUM);


			for (int i = 0; i < TYPE_NUM; i++)
			{
				maxHausdorrf[i] = 0;
				avgHausdorrf[i] = 0;

				for (int t = 0; t < Param[i].size() - 1; t++)
				{
					Hausdorrf[i].push_back(calculateMaxDisBetweenCrvLen(crvs[i], Param[i][t], Param[i][t + 1]));

					if (Hausdorrf[i][t] > maxHausdorrf[i])
						maxHausdorrf[i] = Hausdorrf[i][t];
					avgHausdorrf[i] += Hausdorrf[i][t];
				}

				avgHausdorrf[i] /= (POINTS_NUM - 1);

			}
			*/


			/*
			//计算曲线和折线每一段的长度
			std::vector<std::vector<double>> CurLen;
			CurLen.resize(TYPE_NUM);
			std::vector<double> LinLen;
			LinLen.resize(POINTS_NUM - 1);
			double totalCurlen;
			std::vector<std::vector<double>>  totalCurLens; //每一段贝塞尔曲线的长度
			totalCurLens.resize(TYPE_NUM);
			b_CtrlPoints.resize(TYPE_NUM);
			b_nb.resize(TYPE_NUM);
			vector<vector<double>> b_Qweight;
			std::vector<double> tmp_weight;

			std::vector<double> Crv_Turning_Num;

			Crv_Turning_Num.resize(TYPE_NUM);


			for (int i = 0; i < POINTS_NUM; i++)
			{
				tmp_weight.push_back(1);
			}


			for (int i = 0; i < TYPE_NUM; i++)
			{
				CurLen[i].resize(POINTS_NUM - 1);
				for (int j = 1; j < POINTS_NUM; j++)
				{

					LinLen[j - 1] = calculateLineLength(pts[j - 1], pts[j]);
					CurLen[i][j - 1] = XinPuSenCurveLength(Param[i][j - 1], Param[i][j], crvs[i], 1e-5, 1000000);

					//calculateCurveLength(crvs[i], Param[i][j - 1], Param[i][j], LinLen[j - 1], CurLen[i][j - 1], 1);

				}
			}
			*/


			/*
			//根据曲线和折线长度计算对应的标准
			std::vector<double> DiffOfCur;  // 存放每种方法的曲线和折线长度差值的平均值
			std::vector<double> DiffRioOfCur; //存放每种方法的曲线和折线长度比例的平均值
			std::vector<double> maxDiffRioOfCur; //存放每种方法曲线和折线长度比例的最大值
			for (int i = 0; i < TYPE_NUM; i++)
			{
				double sum = 0;
				double ratiosum = 0;
				double maxratio = CurLen[i][0] / LinLen[0];
				for (int j = 1; j < POINTS_NUM; j++)
				{
					//calculateCurveLength(crvs[i], Param[i][j - 1], Param[i][j], linLen, curLen);
					sum += fabs(CurLen[i][j - 1] - LinLen[j - 1]);
					ratiosum += CurLen[i][j - 1] / LinLen[j - 1];

					if (CurLen[i][j - 1] / LinLen[j - 1] > maxratio) {
						maxratio = CurLen[i][j - 1] / LinLen[j - 1];
					}

				}
				DiffOfCur.push_back(sum / (POINTS_NUM - 1));
				DiffRioOfCur.push_back(ratiosum / (POINTS_NUM - 1));
				maxDiffRioOfCur.push_back(maxratio);
			}
			*/


			/*计算曲线上的最大曲率
			std::vector<double> CrvmaxCurvature;
			for (int i = 0; i < TYPE_NUM; i++)
			{
				double maxCurvature = calculateCurveMaxCurvature(crvs[i]);
				CrvmaxCurvature.push_back(maxCurvature);
			}*/


			/*计算折线上的拐点个数
			std::vector<double> DiffOfTuringNum;
			int LineTurningNum = calculateLineTurningPointNum(pts, LineTuringFlag);
			if (k == draw_id) {
				for (int i = 0; i < LineTuringFlag.size(); i++)
				{
					g_LineTuringFlag.push_back(LineTuringFlag[i]);

				}
			}*/

			/*计算曲线上的拐点个数
			CurveTuringPoints.resize(TYPE_NUM);
			if (k == draw_id) {
				g_CurveTuringPoints.resize(TYPE_NUM);
			}
			std::vector<double> DiffOfTuringNumRatio;
			for (int i = 0; i < TYPE_NUM; i++)
			{

				double CurveTuringNum = calculateCurveTurningPointNum(crvs[i], CurveTuringPoints[i]);

				if (k == draw_id) {
					for (size_t j = 0; j < CurveTuringPoints[i].size(); j++)
					{
						g_CurveTuringPoints[i].push_back(CurveTuringPoints[i][j]);
					}
				}

				DiffOfTuringNum.push_back(abs(CurveTuringNum - LineTurningNum));

				DiffOfTuringNumRatio.push_back((CurveTuringNum + 1) / (LineTurningNum + 1));//加1是为了防止分母为0的情况
			}*/

			/*
			//计算每种方法的各段曲线长度相对于最长段曲线的比例
			std::vector<std::vector<double>> c_rst;
			c_rst.resize(TYPE_NUM);
			double maxCrvLen;
			for (int i = 0; i < TYPE_NUM; i++)
			{
				maxCrvLen = CurLen[i][0];
				for (int j = 1; j < POINTS_NUM; j++)
				{
					if (CurLen[i][j - 1] > maxCrvLen) {
						maxCrvLen = CurLen[i][j - 1];
					}
				}
				for (int j = 1; j < POINTS_NUM; j++)
				{
					c_rst[i].push_back(CurLen[i][j - 1] / maxCrvLen);
				}

			}

			//计算每种方法曲线长度比和折线长度比的差值的均值
			std::vector<double> C_L_Ratio;
			C_L_Ratio.resize(TYPE_NUM);

			for (int i = 0; i < TYPE_NUM; i++)
			{
				C_L_Ratio[i] = 0;
				for (int j = 0; j < l_rst.size(); j++) {
					C_L_Ratio[i] += fabs(l_rst[j] - c_rst[i][j]) / l_rst.size();
				}

			}
			*/

			//评判的最终标准
			/*
			vector<vector<double>> correctHausdorrf; //用折线长度修正的豪斯多夫距离
			vector<double> avgcorrectHausdorrf;//折线长度修正后的豪斯多夫距离平均值

			correctHausdorrf.resize(TYPE_NUM);
			avgcorrectHausdorrf.resize(TYPE_NUM);

			for (int i = 0; i < TYPE_NUM; i++)
			{
				avgcorrectHausdorrf[i] = 0;
				for (int j = 0; j < Hausdorrf[i].size(); j++)
				{
					correctHausdorrf[i].push_back(Hausdorrf[i][j] / LinLen[j]);
					avgcorrectHausdorrf[i] += correctHausdorrf[i][j];
				}
				avgcorrectHausdorrf[i] /= correctHausdorrf[i].size();
			}
			*/


			vector<double> metrics_2013;
			metrics_2013.resize(TYPE_NUM);
			for (int i = 0; i < TYPE_NUM; i++)
			{
				metrics_2013[i] = 0;
				for (int j = 1; j < POINTS_NUM; j++)
				{
					metrics_2013[i] += calculate2013Criteria(crvs[i], Param[i][j - 1], Param[i][j]);
				}
				metrics_2013[i] /= (POINTS_NUM - 1);
			}

			//double Wmin = avgHausdorrf[2];
			double Wmin = metrics_2013[3];

			int min_i = 3;
			for (int i = 3; i < TYPE_NUM; i++) {
				//double W = avgHausdorrf[i];
				double W = metrics_2013[i];
				Cri_W[i] = W;
				if (Wmin > W) {
					Wmin = W;
					min_i = i;
				}
			}
			type_num[min_i]++;


			FeatureLabelData.push_back(min_i + 1);
			std::ofstream OutFile;
			OutFile.open(FeatureLableFileName + "/" + name + ".txt");

			if (!OutFile.is_open()) {
				cout << "OutFile cann't be open." << endl;
				exit(0);
			}

			for (int i = 0; i < pts.size(); i++)
			{
				if (i != 0)
					OutFile << "\n";

				OutFile << pts[i].x << " " << pts[i].y;
			}

			OutFile << "\n";

			for (int i = 0; i < FeatureLabelData.size(); i++)
			{
				if (i != 0)
					OutFile << " ";
				OutFile << FeatureLabelData[i];
			}
			OutFile << "\n";
			OutFile.close();

		}

		float seconds = float(clock() - begin_time) / 1000;
		std::cout << seconds << std::endl;

		int flag_num = 0;

		if (GenerateTrainData(FeatureLableFileName, num, SaveDataFileName, TrainDataNum, TestDataNum, train_num, test_num, POINTS_NUM))
			break;

		std::cout << "\n********************\n" << std::endl;
		GenerateDataDirect(num, POINTS_NUM, flag_num);

	}

	for (int i = 0; i < TYPE_NUM; i++)
	{
		std::cout << type_num[i] << " " << std::endl;
	}



	/*
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(1500, 800);
	glutInitWindowPosition(200, 200);
	glutCreateWindow("三次B样条曲线插值");

	init();
	glutReshapeFunc(Reshape);
	glutKeyboardFunc(keyboard);

	if (draw_mode == 1)
		glutDisplayFunc(Display1);  //方案1：平铺
	else if (draw_mode == 2)
		glutDisplayFunc(Display2);  //方案2：叠加
	else
		glutDisplayFunc(Display3);

	glutMainLoop();
	*/

	return(0);

}
