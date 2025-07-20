#pragma once
#include <glm/glm.hpp>
#include <math.h>
#include <cmath>
#include "tinynurbs.h"
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#define MIN 0.001
#define PI 3.1415926
#define eps 10e-8

using namespace std;

double calculateDerVariance(tinynurbs::Curve<double> crv, double u1, double u2);

/*�������������ʼ���*/
double calculateCurvePointCurvature(tinynurbs::Curve<double> crv, double u);
//template <typename T>
//T calculateCurvePointCurvature(tinynurbs::Curve<T> crv, T u) {
//	typedef glm::vec<3, T, glm::defaultp> vec3T;
//	std::vector<vec3T> derivateData = tinynurbs::curveDerivatives(crv, 2, u);
//
//	T curvature = sqrt(pow((derivateData[1][0] * derivateData[2][1] - derivateData[2][0] * derivateData[1][1]), 2)) /
//		pow(sqrt(derivateData[1][0] * derivateData[1][0] + derivateData[1][1] * derivateData[1][1]), 3);
//
//	return curvature;
//}

/*���������߳��ȼ���: ��ɢ��ʽ���� linLenΪ��Ӧ���߳��� curLenΪ��Ӧ���߳���*/
void calculateCurveLengthLiSan(tinynurbs::Curve<double> crv, double u1, double u2, double& linLen, double& curLen, int m);

/*���������߳��ȼ���: ��˹������ֵ���ּ��� linLenΪ��Ӧ���߳��� curLenΪ��Ӧ���߳���*/
void calculateCurveLength(tinynurbs::Curve<double> crv, double u1, double u2, double& linLen, double& curLen, int m);

/*�������߳��ȼ���: ��˹������ֵ���ּ��� linLenΪ��Ӧ���߳��� curLenΪ��Ӧ���߳���*/
void calculateRationalCurveLength(tinynurbs::RationalCurve<double> crv, double u1, double u2, double& linLen, double& curLen, int m);

/*����Ӧ����ɭ��ֵ���ּ������߳���*/
double XinPuSenCurveLength(double u1, double u2, tinynurbs::Curve<double> crv, double tol, int recursive_num);

/*����Ӧ����ɭ��ֵ���ּ����������߳���*/
double XinPuSenRationalCurveLength(double u1, double u2, tinynurbs::RationalCurve<double> crv, double tol, int recursive_num);

/*�������߳���*/
double calculateLineLength(glm::dvec3 p1, glm::dvec3 p2);


/*���������߹�˳�����: ���ʵ�ƽ�������ߵĻ��� ��˹������ֵ���ּ��� crvΪ�������� mΪ�ֳɵĶ���*/
double calculateCurveCurvatureIntegral(tinynurbs::Curve<double> crv, int m); 

/*�����˳�����: ���ʵ�ƽ�������ߵĻ��� ��˹������ֵ���ּ��� crvΪ�������� mΪ�ֳɵĶ���*/
double calculateRationalCurveCurvatureIntegral(tinynurbs::RationalCurve<double> crv, int m); 

/*���������ϵĹյ㣺��ɢ����*/
int calculateCurveTurningPointNum(tinynurbs::Curve<double> crv, std::vector<glm::dvec3>& CurveTuringPoints);

/*����һ�����α������Ϲյ㣺��������  �������Ϊ���Ƶ�����*/
int calculateCurveTurningNum(std::vector<glm::dvec3> P);

/*����һ�����α����������Ϲյ����������Ԫ�жϷ��� �������Ϊ���Ƶ�����*/
int calculateCurveTurningNumLiu(std::vector<glm::dvec3> P);

/*���������ϵĹյ�*/
int calculateLineTurningPointNum(std::vector<glm::dvec3>& pts, std::vector<bool>& LineTuringFlag);

/*�����������ߺ�˹������: ��ɢ����*/
double calculateMaxDisBetweenCrvLen(tinynurbs::Curve<double> crv, double u1, double u2);
//template <typename T>
//T calculateMaxDisBetweenCrvLen(tinynurbs::Curve<T> crv, T u1, T u2) {
//	T A, B, C;
//	typedef glm::vec<3, T, glm::defaultp> vec3T;
//	vec3T data1 = tinynurbs::curvePoint(crv, u1);
//	vec3T data2 = tinynurbs::curvePoint(crv, u2);
//
//	A = (data1.y - data2.y) / (data1.x - data2.x);
//	B = -1;
//	C = data1.y - A * data1.x;
//	T s_num = 50;
//	T delta = (u2 - u1) / s_num;
//	T u = u1;
//	T tmpdis;
//	T maxdis = 0;
//	std::vector<vec3T> derivateData;
//	T s;
//
//	for (int i = 1; i <= s_num; i++)
//	{
//		vec3T data3 = tinynurbs::curvePoint(crv, u + delta / 2);
//		tmpdis = abs(A * data3.x + B * data3.y + C) / sqrt(A * A + B * B);
//		if (tmpdis > maxdis)
//			maxdis = tmpdis;
//		u += delta;
//	}
//	return maxdis;
//}

/*�����������ߺ�˹������: ��ɢ����*/
double calculateMaxDisBetweenRationalCrvLen(tinynurbs::RationalCurve<double> crv, double u1, double u2); 

/*һԪ���ζ���ʽ*/
double CubicFun(double a, double b, double c, double d, double u);
/*����һ�����α��������߸�һ�����ߵĺ�˹������
*���������߿��Ƶ�����
* ���߲�����Χ
* �������˵�����
*/
double CalculateHausdorffDistanceBetweenBL(std::vector<glm::dvec3> C, double u1, double u2, glm::dvec3 p1, glm::dvec3 p2);

/*����ÿһ�ε����ݵ�ĺ�˹������
* ���б��������߿��Ƶ�
* �������ݵ�
* ����
* �ڵ�
*/
vector<double>  CalculateHausdorffDistance(vector<vector<glm::dvec3>> C, vector<glm::dvec3> P, vector<double> Param, vector<double> Knots);

/* �ڵ�ϸ��*/
void DecomposeCurve(int n, int p, std::vector<double> knots, std::vector<double> Pweight, std::vector<glm::dvec3> Pw, std::vector<std::vector<double>>& Qweight, int& nb, std::vector<std::vector<glm::dvec3>>& Qw);

/*�ڵ�ϸ��֮����㲢��������ƽ�������߻��ֺ������ܳ���*/
void DecomposeCurveToSave(int n, int p, std::vector<double> knots, std::vector<double> Pweight, std::vector<glm::dvec3> Pw,
	std::vector<std::vector<double>>& Qweight, int& nb, std::vector<std::vector<glm::dvec3>>& Qw, double& resintegral); 

/*�����ϵ�������ʼ��㣺��ɢ����*/
double calculateCurveMaxCurvature(tinynurbs::Curve<double> crv);

/*������ĳһ�������ϵ��������: ��ɢ����*/
double calculateMaxCurvatureOnPartCurve(tinynurbs::Curve<double> crv, double u1, double u2);
//template <typename T>
//T calculateMaxCurvatureOnPartCurve(tinynurbs::Curve<T> crv, T u1, T u2) {
//
//	//double sum = 0;
//	typedef glm::vec<3, T, glm::defaultp> vec3T;
//	T s_num = 50;
//	T delta = (u2 - u1) / s_num;
//	T u = u1;
//	T curvature;
//	T maxcurvature = 0;
//	std::vector<vec3T> derivateData;
//	T s;
//	for (int i = 1; i <= s_num; i++)
//	{
//		curvature = calculateCurvePointCurvature(crv, u + delta / 2);
//		if (curvature > maxcurvature)
//			maxcurvature = curvature;
//
//		u += delta;
//	}
//	return maxcurvature;
//}

/*2013�������е�����ָ��*/
double calculate2013Criteria(tinynurbs::Curve<double> crv, double u1, double u2);
//template <typename T>
//T calculate2013Criteria(tinynurbs::Curve<T> crv, T u1, T u2) {
//
//	T maxdis = calculateMaxDisBetweenCrvLen(crv, u1, u2);
//	T maxcur = calculateMaxCurvatureOnPartCurve(crv, u1, u2);
//	return (1 + maxcur) * maxdis;
//}

double calculateModifiedCriteria(tinynurbs::Curve<double> crv, double u1, double u2);

/*��ֵ���ֲ���*/
long double fun(long double x);
long double calculatGaussLengendre(long double u1, long double u2);
long double calculateXinpusen(long double u1, long double u2); 