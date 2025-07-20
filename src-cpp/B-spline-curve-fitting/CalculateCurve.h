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

/*非有理曲线曲率计算*/
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

/*非有理曲线长度计算: 离散方式计算 linLen为对应折线长度 curLen为对应曲线长度*/
void calculateCurveLengthLiSan(tinynurbs::Curve<double> crv, double u1, double u2, double& linLen, double& curLen, int m);

/*非有理曲线长度计算: 高斯勒让数值积分计算 linLen为对应折线长度 curLen为对应曲线长度*/
void calculateCurveLength(tinynurbs::Curve<double> crv, double u1, double u2, double& linLen, double& curLen, int m);

/*有理曲线长度计算: 高斯勒让数值积分计算 linLen为对应折线长度 curLen为对应曲线长度*/
void calculateRationalCurveLength(tinynurbs::RationalCurve<double> crv, double u1, double u2, double& linLen, double& curLen, int m);

/*自适应辛普森数值积分计算曲线长度*/
double XinPuSenCurveLength(double u1, double u2, tinynurbs::Curve<double> crv, double tol, int recursive_num);

/*自适应辛普森数值积分计算有理曲线长度*/
double XinPuSenRationalCurveLength(double u1, double u2, tinynurbs::RationalCurve<double> crv, double tol, int recursive_num);

/*计算折线长度*/
double calculateLineLength(glm::dvec3 p1, glm::dvec3 p2);


/*非有理曲线光顺项计算: 曲率的平方沿曲线的积分 高斯勒让数值积分计算 crv为曲线数据 m为分成的段数*/
double calculateCurveCurvatureIntegral(tinynurbs::Curve<double> crv, int m); 

/*有理光顺项计算: 曲率的平方沿曲线的积分 高斯勒让数值积分计算 crv为曲线数据 m为分成的段数*/
double calculateRationalCurveCurvatureIntegral(tinynurbs::RationalCurve<double> crv, int m); 

/*计算曲线上的拐点：离散方法*/
int calculateCurveTurningPointNum(tinynurbs::Curve<double> crv, std::vector<glm::dvec3>& CurveTuringPoints);

/*计算一段三次贝塞尔上拐点：解析计算  传入参数为控制点坐标*/
int calculateCurveTurningNum(std::vector<glm::dvec3> P);

/*计算一段三次贝塞尔曲线上拐点个数：刘鼎元判断方法 传入参数为控制点坐标*/
int calculateCurveTurningNumLiu(std::vector<glm::dvec3> P);

/*计算折线上的拐点*/
int calculateLineTurningPointNum(std::vector<glm::dvec3>& pts, std::vector<bool>& LineTuringFlag);

/*计算无理曲线豪斯多夫距离: 离散方法*/
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

/*计算有理曲线豪斯多夫距离: 离散方法*/
double calculateMaxDisBetweenRationalCrvLen(tinynurbs::RationalCurve<double> crv, double u1, double u2); 

/*一元三次多项式*/
double CubicFun(double a, double b, double c, double d, double u);
/*计算一段三次贝塞尔曲线跟一段折线的豪斯多夫距离
*贝塞尔曲线控制点坐标
* 曲线参数范围
* 折线两端点坐标
*/
double CalculateHausdorffDistanceBetweenBL(std::vector<glm::dvec3> C, double u1, double u2, glm::dvec3 p1, glm::dvec3 p2);

/*计算每一段的数据点的豪斯多夫距离
* 所有贝塞尔曲线控制点
* 所有数据点
* 参数
* 节点
*/
vector<double>  CalculateHausdorffDistance(vector<vector<glm::dvec3>> C, vector<glm::dvec3> P, vector<double> Param, vector<double> Knots);

/* 节点细化*/
void DecomposeCurve(int n, int p, std::vector<double> knots, std::vector<double> Pweight, std::vector<glm::dvec3> Pw, std::vector<std::vector<double>>& Qweight, int& nb, std::vector<std::vector<glm::dvec3>>& Qw);

/*节点细化之后计算并返回曲率平方沿曲线积分和曲线总长度*/
void DecomposeCurveToSave(int n, int p, std::vector<double> knots, std::vector<double> Pweight, std::vector<glm::dvec3> Pw,
	std::vector<std::vector<double>>& Qweight, int& nb, std::vector<std::vector<glm::dvec3>>& Qw, double& resintegral); 

/*曲线上的最大曲率计算：离散方法*/
double calculateCurveMaxCurvature(tinynurbs::Curve<double> crv);

/*计算在某一段曲线上的最大曲率: 离散方法*/
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

/*2013年论文中的评判指标*/
double calculate2013Criteria(tinynurbs::Curve<double> crv, double u1, double u2);
//template <typename T>
//T calculate2013Criteria(tinynurbs::Curve<T> crv, T u1, T u2) {
//
//	T maxdis = calculateMaxDisBetweenCrvLen(crv, u1, u2);
//	T maxcur = calculateMaxCurvatureOnPartCurve(crv, u1, u2);
//	return (1 + maxcur) * maxdis;
//}

double calculateModifiedCriteria(tinynurbs::Curve<double> crv, double u1, double u2);

/*数值积分测试*/
long double fun(long double x);
long double calculatGaussLengendre(long double u1, long double u2);
long double calculateXinpusen(long double u1, long double u2); 