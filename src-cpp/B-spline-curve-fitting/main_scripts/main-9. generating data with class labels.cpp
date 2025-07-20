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

/*opengl����nurbs��������ͼ���ʱ�õ�����*/
vector<bool> g_LineTuringFlag;  //�����ϵĹյ���
vector<vector<glm::dvec3>> g_CurveTuringPoints;//�����ϵĹյ�����
vector<glm::dvec3> g_pts; //���ݵ�
vector<vector<glm::dvec3>> g_CtrlPoints; // ���Ƶ�  ��Ӧ��ϵ�� uniform chord centripetal universal correctchord RefinedCentripttal
vector<vector<double>>  g_Knots; //�ڵ�ʸ��
vector<vector<vector<glm::dvec3>>> b_CtrlPoints; //�ڵ�ϸ��֮��ı��������ߵĿ��Ƶ�
vector<vector<glm::dvec3>> knots_pts; //�ڵ�ʸ����Ӧ�ĵ�
vector<int>  b_nb; //ÿ��nurbs�ڵ�ϸ����ı��������ߵĸ���
double Cri_W[TYPE_NUM];  //����opengl��Ӧnurbs������ָ��
int type_num[TYPE_NUM] = { 0 };//��������������ÿ�ַ�����ѡ���Ĵ���
vector<vector<double>> b_hausdorrf; //����ڵ�ϸ����ı���������ĺ�˹������
vector<vector<double>> p_hausdorrf; //������ɢ�����ĺ�˹������

/*opengl����nurbs�����õ�����*/
GLUnurbsObj* theNurb;
GLfloat ctrlpoints[COTROL_NUM][3];
GLfloat knots[KNOTS_NUM];
GLfloat dataPoints[POINTS_NUM][3];//����ϵ����ݵ�
GLfloat knotsPoints[TYPE_NUM][POINTS_NUM - 2][3]; // �ڵ�ʸ����Ӧ�ĵ� - ���Խڵ�ϸ���ͽڵ�ʸ���Ķ�Ӧ��ϵ
char type_name[TYPE_NUM][20] = { "Uniform ", "Chord ", "Centripetal ", "Universal ", "CorrectChord ", "RefinedCentripttal" };
double max_length, max_width, min_length, min_width, gap_width, gap_length;  //���ڴ��ڳ���
bool draw_flag[TYPE_NUM] = { true }; // ÿ�ַ�����Ӧnurbs�Ƿ���ƿ���
bool draw_points_flag = true; //���ݵ���ƿ���
bool draw_knots_flag = true; //�ڵ�ʸ�����ƿ���
bool draw_line_flag = true; //���߻��ƿ���
GLfloat testdata[TYPE_NUM][3]; //������opengl����



int main(int argc, char** argv)
{
	int id;

	//int draw_id = 0;
	//int draw_mode = 1;

	int	num; //ÿ��ԭʼ�������ɵ����� // Quantity of raw data generated per round
	int	train_num; //ÿ����ǩ��ѵ�������� // Training data quantity per label
	int test_num; // ÿ����ǩ�Ĳ��������� // Test data quantity per label
	string	FeatureLableFileName; //�����ͱ�ǩ�ļ������� // Feature and label folder name
	string	DataFileName; //ԭʼ�����ļ������� // Raw data folder name
	string	SaveDataFileName;// ���մ�ñ�ǩ�����������ļ������� // Generated labeled data folder name


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


			vector<glm::dvec3> pts; //Ϊ������ͼ��㷽��
			vector<vector<glm::dvec3>> CtrlPoints; // ���Ƶ�  ��Ӧ��ϵ�� uniform chord centripetal universal correctchord RefinedCentripttal
			vector<vector<double>>  Knots; //�ڵ�
			vector<vector<double>> Param; //����
			vector<tinynurbs::Curve<double>> crvs; //��������
			vector<vector<glm::dvec3>> CurveTuringPoints;//��������еĹյ�
			vector<bool> LineTuringFlag;  //��������еĹյ���

			vector<double> FeatureLabelData; //�����ͱ�ǩ������
			vector<double> Matrix; //����ָ���ֵ

			/*�����Ӧ�ļ����ı��*/
			std::string name;
			name = NumToStr(k);

			//name = NumToStr(4) + NumToStr(k);


			/*���ļ��ж�ȡ��Ӧ������*/
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

			/*���ֲ����������Ĳ�ֵ����*/
			CtrlPoints.clear();
			Knots.clear();
			Param.clear();
			CtrlPoints.resize(TYPE_NUM);
			Knots.resize(TYPE_NUM);
			Param.resize(TYPE_NUM);

			/*���� Uniform �Ĳ�ֵ  */
			UniformInterp(pts, 3, Knots[0], Param[0]);
			GlobalInterp(pts, 3, Knots[0], Param[0], CtrlPoints[0]);

			/*���� Chord �Ĳ�ֵ */
			ChordInterp(pts, 3, Knots[1], Param[1]);
			GlobalInterp(pts, 3, Knots[1], Param[1], CtrlPoints[1]);

			/*���� Centripetal �Ĳ�ֵ */
			CentripetalInterp(pts, 3, Knots[2], Param[2]);
			GlobalInterp(pts, 3, Knots[2], Param[2], CtrlPoints[2]);

			/*���� Universal �Ĳ�ֵ */
			UniversalInterp(pts, 3, Knots[3], Param[3]);
			GlobalInterp(pts, 3, Knots[3], Param[3], CtrlPoints[3]);

			/*���� CorrectChord �Ĳ�ֵ */
			CorrectChordInterp(pts, 3, Knots[4], Param[4]);
			GlobalInterp(pts, 3, Knots[4], Param[4], CtrlPoints[4]);

			/*���� RefinedCentripetal �Ĳ�ֵ*/
			RefinedCentripetalInterp(pts, 3, Knots[5], Param[5]);
			GlobalInterp(pts, 3, Knots[5], Param[5], CtrlPoints[5]);

			/*����ڵ�ʸ���Ϳ��Ƶ㵽crvs�����������*/
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



			//���б�׼�ļ���
			/*
			//�������ʵ�ƽ�������ߵĻ���
			std::vector<double> totalCrvIntegrals; //�÷ֶα�������ֵ���ּ�����
			std::vector<double> CrvIntegrals;  //ֱ��������ֵ���ּ���Ľ��
			totalCrvIntegrals.resize(TYPE_NUM);
			for (int i = 0; i < TYPE_NUM; i++)
			{
				double CurvatureIntegral = calculateCurveCurvatureIntegral(crvs[i], 20);
				CrvIntegrals.push_back(CurvatureIntegral);
			}
			*/



			/*��˹������ļ���*/
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
			//�������ߺ�����ÿһ�εĳ���
			std::vector<std::vector<double>> CurLen;
			CurLen.resize(TYPE_NUM);
			std::vector<double> LinLen;
			LinLen.resize(POINTS_NUM - 1);
			double totalCurlen;
			std::vector<std::vector<double>>  totalCurLens; //ÿһ�α��������ߵĳ���
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
			//�������ߺ����߳��ȼ����Ӧ�ı�׼
			std::vector<double> DiffOfCur;  // ���ÿ�ַ��������ߺ����߳��Ȳ�ֵ��ƽ��ֵ
			std::vector<double> DiffRioOfCur; //���ÿ�ַ��������ߺ����߳��ȱ�����ƽ��ֵ
			std::vector<double> maxDiffRioOfCur; //���ÿ�ַ������ߺ����߳��ȱ��������ֵ
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


			/*���������ϵ��������
			std::vector<double> CrvmaxCurvature;
			for (int i = 0; i < TYPE_NUM; i++)
			{
				double maxCurvature = calculateCurveMaxCurvature(crvs[i]);
				CrvmaxCurvature.push_back(maxCurvature);
			}*/


			/*���������ϵĹյ����
			std::vector<double> DiffOfTuringNum;
			int LineTurningNum = calculateLineTurningPointNum(pts, LineTuringFlag);
			if (k == draw_id) {
				for (int i = 0; i < LineTuringFlag.size(); i++)
				{
					g_LineTuringFlag.push_back(LineTuringFlag[i]);

				}
			}*/

			/*���������ϵĹյ����
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

				DiffOfTuringNumRatio.push_back((CurveTuringNum + 1) / (LineTurningNum + 1));//��1��Ϊ�˷�ֹ��ĸΪ0�����
			}*/

			/*
			//����ÿ�ַ����ĸ������߳��������������ߵı���
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

			//����ÿ�ַ������߳��ȱȺ����߳��ȱȵĲ�ֵ�ľ�ֵ
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

			//���е����ձ�׼
			/*
			vector<vector<double>> correctHausdorrf; //�����߳��������ĺ�˹������
			vector<double> avgcorrectHausdorrf;//���߳���������ĺ�˹������ƽ��ֵ

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
	glutCreateWindow("����B�������߲�ֵ");

	init();
	glutReshapeFunc(Reshape);
	glutKeyboardFunc(keyboard);

	if (draw_mode == 1)
		glutDisplayFunc(Display1);  //����1��ƽ��
	else if (draw_mode == 2)
		glutDisplayFunc(Display2);  //����2������
	else
		glutDisplayFunc(Display3);

	glutMainLoop();
	*/

	return(0);

}
