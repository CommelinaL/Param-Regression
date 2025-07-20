//#include <glm/glm.hpp>
//#include <math.h>
//#include <cmath>

#include <iostream>
//#include <vector>
//#include <Eigen/Dense>

//#include "tinynurbs.h"

#include "CalculateCurve.h"

using namespace std;

double calculateDerVariance(tinynurbs::Curve<double> crv, double u1, double u2) {
	int s_num = 10000;
	double delta = (u2 - u1) / s_num;
	double u = u1;
	std::vector<glm::dvec3> derivateData;
	double der_sqr, avg = 0, var=0;
	for (int i = 1; i <= s_num; i++)
	{
		std::vector<glm::dvec3> derivateData = tinynurbs::curveDerivatives(crv, 2, u);
		der_sqr = derivateData[1][0] * derivateData[1][0] + derivateData[1][1] * derivateData[1][1];
		var += der_sqr;
		avg += sqrt(der_sqr);
		u += delta;
	}
	var /= s_num;
	avg /= s_num;
	var = var - avg * avg;
	return var;
}

/*非有理曲线曲率计算：参考高数中参数曲线的曲率计算*/
double calculateCurvePointCurvature(tinynurbs::Curve<double> crv, double u) {
	std::vector<glm::dvec3> derivateData = tinynurbs::curveDerivatives(crv, 2, u);

	double curvature = sqrt(pow((derivateData[1][0] * derivateData[2][1] - derivateData[2][0] * derivateData[1][1]), 2)) /
		pow(sqrt(derivateData[1][0] * derivateData[1][0] + derivateData[1][1] * derivateData[1][1]), 3);

	//std::cout << "curvature: " << curvature << std::endl;
	return curvature;
}

/*非有理曲线长度计算: 离散方式计算 linLen为对应折线长度 curLen为对应曲线长度*/
void calculateCurveLengthLiSan(tinynurbs::Curve<double> crv, double u1, double u2, double& linLen, double& curLen, int m) {
	glm::vec3 data1 = tinynurbs::curvePoint(crv, u1);
	glm::vec3 data2 = tinynurbs::curvePoint(crv, u2);

	linLen = sqrt(pow(data1.x - data2.x, 2) + pow(data1.y - data2.y, 2));

	double stride = (u2 - u1) / m;
	u2 = 0;
	curLen = 0;
	for (int k = 0; k < m; k++)
	{
		u1 = u2;
		u2 += stride;
		data1 = tinynurbs::curvePoint(crv, u1);
		data2 = tinynurbs::curvePoint(crv, u2);
		curLen += sqrt(pow(data1.x - data2.x, 2) + pow(data1.y - data2.y, 2));
	}


}

/*非有理曲线长度计算
高斯勒让数值积分计算
linLen为对应折线长度
curLen为对应曲线长度
*/
void calculateCurveLength(tinynurbs::Curve<double> crv, double u1, double u2, double& linLen, double& curLen, int m) {

	glm::vec3 data1 = tinynurbs::curvePoint(crv, u1);
	glm::vec3 data2 = tinynurbs::curvePoint(crv, u2);
	linLen = sqrt(pow(data1.x - data2.x, 2) + pow(data1.y - data2.y, 2));

	double x5[5] = { 0.0, -0.5384693101056831, 0.5384693101056831, -0.9061798459386640, 0.9061798459386640 };

	double w5[5] = { 0.56888888888889, 0.4786286704993665, 0.4786286704993665, 0.2369268850561891, 0.2369268850561891 };

	double x8[8] = { -0.9602898565, -0.7966664774, -0.5255324099, -0.1834346425,
				0.1834346425, 0.5255324099, 0.7966664774, 0.9602898565 };
	double w8[8] = { 0.1012285363, 0.2223810345, 0.3137066459, 0.3626837834,
					0.3626837834,  0.3137066459, 0.2223810345, 0.1012285363 };

	curLen = 0;

	std::vector<glm::dvec3> derivateData;
	int n = 8;

	double stride = (u2 - u1) / m;
	u2 = 0;
	double tmp;
	for (int k = 0; k < m; k++)
	{
		u1 = u2;
		u2 += stride;
		tmp = 0;

		for (int i = 0; i < n; i++) {

			double u = x5[i] * (u2 - u1) / 2.0 + (u1 + u2) / 2.0;
			derivateData = tinynurbs::curveDerivatives(crv, 1, u);

			tmp += w5[i] * sqrt(derivateData[1][0] * derivateData[1][0] + derivateData[1][1] * derivateData[1][1]);

		}
		tmp *= (u2 - u1) / 2;
		curLen += tmp;
	}

}

/*有理曲线长度计算
高斯勒让数值积分计算
linLen为对应折线长度
curLen为对应曲线长度
*/
void calculateRationalCurveLength(tinynurbs::RationalCurve<double> crv, double u1, double u2, double& linLen, double& curLen, int m) {

	glm::vec3 data1 = tinynurbs::curvePoint(crv, u1);
	glm::vec3 data2 = tinynurbs::curvePoint(crv, u2);

	double x[5] = { 0.0, -0.5384693101056831, 0.5384693101056831, -0.9061798459386640, 0.9061798459386640 };

	double w[5] = { 0.56888888888889, 0.4786286704993665, 0.4786286704993665, 0.2369268850561891, 0.2369268850561891 };


	double x8[8] = { -0.9602898565, -0.7966664774, -0.5255324099, -0.1834346425,
					0.1834346425, 0.5255324099, 0.7966664774, 0.9602898565 };
	double w8[8] = { 0.1012285363, 0.2223810345, 0.3137066459, 0.3626837834,
					0.3626837834,  0.3137066459, 0.2223810345, 0.1012285363 };

	linLen = sqrt(pow(data1.x - data2.x, 2) + pow(data1.y - data2.y, 2));

	curLen = 0;
	std::vector<glm::dvec3> derivateData;
	int n = 8;
	double stride = (u2 - u1) / m;
	u2 = 0;
	double tmp;
	for (int k = 0; k < m; k++)
	{
		u1 = u2;
		u2 += stride;
		tmp = 0;

		for (int i = 0; i < n; i++) {

			double u = x8[i] * (u2 - u1) / 2.0 + (u1 + u2) / 2.0;
			derivateData = tinynurbs::curveDerivatives(crv, 1, u);

			tmp += w8[i] * sqrt(derivateData[1][0] * derivateData[1][0] + derivateData[1][1] * derivateData[1][1]);

		}
		tmp *= (u2 - u1) / 2;
		curLen += tmp;
	}
}


/*自适应辛普森数值积分计算曲线长度*/
double XinPuSenCurveLength(double u1, double u2, tinynurbs::Curve<double> crv, double tol, int recursive_num)
{

	double mid = (u1 + u2) / 2;

	std::vector<glm::dvec3> derivateData1;
	std::vector<glm::dvec3> derivateData2;
	std::vector<glm::dvec3> derivateData3;
	glm::vec3 data1 = tinynurbs::curvePoint(crv, u1);
	glm::vec3 data2 = tinynurbs::curvePoint(crv, u2);
	derivateData1 = tinynurbs::curveDerivatives(crv, 1, u1);
	derivateData2 = tinynurbs::curveDerivatives(crv, 1, u2);
	derivateData3 = tinynurbs::curveDerivatives(crv, 1, mid);

	double L = 0.0;
	double L1 = sqrt(pow(data1.x - data2.x, 2) + pow(data1.y - data2.y, 2) );
	
	double d1 = sqrt(pow(derivateData1[1][0], 2) + pow(derivateData1[1][1], 2));
	double d2 = sqrt(pow(derivateData2[1][0], 2) + pow(derivateData2[1][1], 2));
	double d3 = sqrt(pow(derivateData3[1][0], 2) + pow(derivateData3[1][1], 2));

	double L2 = (u2 - u1) / 2 / 3 * (d1 + 4 * d3 + d2);

	if (std::abs(L1 - L2) < tol || recursive_num == 0 ) {
		L = (L1 + L2) / 2;
	}
	else {
		L = XinPuSenCurveLength(u1, mid, crv,  tol, recursive_num - 1) + XinPuSenCurveLength(mid, u2, crv, tol, recursive_num - 1);
	}
	return L;
}


/*自适应辛普森数值积分计算有理曲线长度*/
double XinPuSenRationalCurveLength(double u1, double u2, tinynurbs::RationalCurve<double> crv, double tol, int recursive_num)
{

	double mid = (u1 + u2) / 2;

	std::vector<glm::dvec3> derivateData1;
	std::vector<glm::dvec3> derivateData2;
	std::vector<glm::dvec3> derivateData3;
	glm::vec3 data1 = tinynurbs::curvePoint(crv, u1);
	glm::vec3 data2 = tinynurbs::curvePoint(crv, u2);
	derivateData1 = tinynurbs::curveDerivatives(crv, 1, u1);
	derivateData2 = tinynurbs::curveDerivatives(crv, 1, u2);
	derivateData3 = tinynurbs::curveDerivatives(crv, 1, mid);

	double L = 0.0;
	double L1 = sqrt(pow(data1.x - data2.x, 2) + pow(data1.y - data2.y, 2));

	double d1 = sqrt(pow(derivateData1[1][0], 2) + pow(derivateData1[1][1], 2));
	double d2 = sqrt(pow(derivateData2[1][0], 2) + pow(derivateData2[1][1], 2));
	double d3 = sqrt(pow(derivateData3[1][0], 2) + pow(derivateData3[1][1], 2));

	double L2 = (u2 - u1) / 2 / 3 * (d1 + 4 * d3 + d2);

	if (std::abs(L1 - L2) < tol || recursive_num == 0) {
		L = (L1 + L2) / 2;
	}
	else {
		L = XinPuSenRationalCurveLength(u1, mid, crv, tol, recursive_num - 1) + XinPuSenRationalCurveLength(mid, u2, crv, tol, recursive_num - 1);
	}
	return L;
}



/*计算折线长度*/
double calculateLineLength(glm::dvec3 p1, glm::dvec3 p2) {

	double len = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
	return len;
}




/*曲线上的最大曲率计算：离散方法*/
double calculateCurveMaxCurvature(tinynurbs::Curve<double> crv) {
	double s_num = 10000;
	double delta = 1.0 / s_num;
	double u = 0;
	double curvature;
	double maxcurvature = 0;
	std::vector<glm::dvec3> derivateData;
	double s;
	for (int i = 1; i <= s_num; i++)
	{
		curvature = calculateCurvePointCurvature(crv, u + delta / 2);
		if (curvature > maxcurvature)
			maxcurvature = curvature;

		u += delta;

	}
	return maxcurvature;

}


/*非有理曲线光顺项计算: 曲率的平方沿曲线的积分
辛普森数值积分计算
crv为曲线数据
m为分成的段数
*/
double calculateCurveCurvatureIntegral(tinynurbs::Curve<double> crv, int m) {

	int n = 8;
	double u = 0;

	double x[5] = { 0.0, -0.5384693101056831, 0.5384693101056831, -0.9061798459386640, 0.9061798459386640 };

	double w[5] = { 0.56888888888889, 0.4786286704993665, 0.4786286704993665, 0.2369268850561891, 0.2369268850561891 };

	long double x8[8] = { -0.9602898565, -0.7966664774, -0.5255324099, -0.1834346425,
				0.1834346425, 0.5255324099, 0.7966664774, 0.9602898565 };
	long double w8[8] = { 0.1012285363, 0.2223810345, 0.3137066459, 0.3626837834,
					0.3626837834,  0.3137066459, 0.2223810345, 0.1012285363 };
	double sum = 0;
	double tmp;
	double u1, u2;

	std::vector<glm::dvec3> derivateData;

	u2 = 0;
	for (int k = 0; k < m; k++)
	{
		u1 = u2;
		u2 += 1.0 / m;
		tmp = 0;
		for (int i = 0; i < n; i++)
		{

			u = x8[i] * (u2 - u1) / 2.0 + (u1 + u2) / 2.0;
			derivateData = tinynurbs::curveDerivatives(crv, 2, u);

			tmp += w8[i] * (pow((derivateData[1][0] * derivateData[2][1] - derivateData[2][0] * derivateData[1][1]), 2)
				/ pow(sqrt(derivateData[1][0] * derivateData[1][0] + derivateData[1][1] * derivateData[1][1]), 5));

		}
		tmp *= (u2 - u1) / 2;

		sum += tmp;
	}


	return sum;

}

/*有理光顺项计算: 曲率的平方沿曲线的积分
辛普森数值积分计算
crv为曲线数据
m为分成的段数
*/

double calculateRationalCurveCurvatureIntegral(tinynurbs::RationalCurve<double> crv, int m) {


	int n = 8;
	double u = 0;

	double x[5] = { 0.0, -0.5384693101056831, 0.5384693101056831, -0.9061798459386640, 0.9061798459386640 };

	double w[5] = { 0.56888888888889, 0.4786286704993665, 0.4786286704993665, 0.2369268850561891, 0.2369268850561891 };

	long double x8[8] = { -0.9602898565, -0.7966664774, -0.5255324099, -0.1834346425,
				0.1834346425, 0.5255324099, 0.7966664774, 0.9602898565 };
	long double w8[8] = { 0.1012285363, 0.2223810345, 0.3137066459, 0.3626837834,
					0.3626837834,  0.3137066459, 0.2223810345, 0.1012285363 };
	double sum = 0;
	double tmp;
	double u1, u2;

	std::vector<glm::dvec3> derivateData;

	u2 = 0;
	for (int k = 0; k < m; k++)
	{
		u1 = u2;
		u2 += 1.0 / m;
		tmp = 0;
		for (int i = 0; i < n; i++)
		{

			u = x8[i] * (u2 - u1) / 2.0 + (u1 + u2) / 2.0;
			derivateData = tinynurbs::curveDerivatives(crv, 2, u);

			tmp += w8[i] * (pow((derivateData[1][0] * derivateData[2][1] - derivateData[2][0] * derivateData[1][1]), 2)
				/ pow(sqrt(derivateData[1][0] * derivateData[1][0] + derivateData[1][1] * derivateData[1][1]), 5));

		}
		tmp *= (u2 - u1) / 2;

		sum += tmp;
	}
	return sum;
}


/*计算曲线上的拐点：离散方法*/
int calculateCurveTurningPointNum(tinynurbs::Curve<double> crv, std::vector<glm::dvec3>& CurveTuringPoints) {

	int num = 0;
	double s_num = 8000;
	double delta = 1 / s_num;
	double u = 0;
	std::vector<double> curvatures;
	std::vector<glm::dvec3> derivateData1 = tinynurbs::curveDerivatives(crv, 2, u);
	curvatures.push_back(sqrt(pow((derivateData1[1][0] * derivateData1[2][1] - derivateData1[2][0] * derivateData1[1][1]), 2)) /
		pow(sqrt(derivateData1[1][0] * derivateData1[1][0] + derivateData1[1][1] * derivateData1[1][1]), 3));

	for (int i = 0; i < s_num; i++)
	{
		u += delta;
		std::vector<glm::dvec3> derivateData2 = tinynurbs::curveDerivatives(crv, 2, u);
		curvatures.push_back(sqrt(pow((derivateData2[1][0] * derivateData2[2][1] - derivateData2[2][0] * derivateData2[1][1]), 2)) /
			pow(sqrt(derivateData2[1][0] * derivateData2[1][0] + derivateData2[1][1] * derivateData2[1][1]), 3));
		if ((derivateData1[1][0] * derivateData1[2][1] - derivateData1[2][0] * derivateData1[1][1])
			* (derivateData2[1][0] * derivateData2[2][1] - derivateData2[2][0] * derivateData2[1][1]) < 0) {
			num++;
		}
		derivateData1 = derivateData2;
	}
	return num;
}

/*计算折线上的拐点*/


int calculateLineTurningPointNum(std::vector<glm::dvec3>& pts, std::vector<bool>& LineTuringFlag) {

	int num = 0;
	LineTuringFlag.push_back(false);
	for (int i = 1; i < pts.size() - 1; i++)
	{
		double deltx1 = pts[i].x - pts[i - 1].x;
		double delty1 = pts[i].y - pts[i - 1].y;
		double deltx2 = pts[i + 1].x - pts[i].x;
		double delty2 = pts[i + 1].y - pts[i].y;
		//std::cout << "delt: " << deltx1 << " " << delty1 << " " << deltx2 << " " << delty2 << std::endl;
		if ((deltx1 * deltx2) > 0 && (delty1 * delty2) > 0) {
			//std::cout << "delt: " << deltx1 << " " << delty1 << " " << deltx2 << " " << delty2 << std::endl;
			//std::cout << "delt: " << deltx1 * deltx2 << " " << delty1 * delty2  << std::endl;
			LineTuringFlag.push_back(false);
		}
		else
		{
			num++;
			LineTuringFlag.push_back(true);

		}

	}
	LineTuringFlag.push_back(false);
	return num;
}

/*计算无理曲线豪斯多夫距离: 离散方法*/
double calculateMaxDisBetweenCrvLen(tinynurbs::Curve<double> crv, double u1, double u2) {
	//double sum = 0;
	double A, B, C;
	glm::vec3 data1 = tinynurbs::curvePoint(crv, u1);
	glm::vec3 data2 = tinynurbs::curvePoint(crv, u2);

	A = (data1.y - data2.y) / (data1.x - data2.x);
	B = -1;
	C = data1.y - A * data1.x;
	double s_num = 10000;
	double delta = (u2 - u1) / s_num;
	double u = u1;
	double tmpdis;
	double maxdis = 0;
	std::vector<glm::dvec3> derivateData;
	double s;

	for (int i = 1; i <= s_num; i++)
	{
		glm::vec3 data3 = tinynurbs::curvePoint(crv, u + delta / 2);
		tmpdis = fabs(A * data3.x + B * data3.y + C) / sqrt(A * A + B * B);
		if (tmpdis > maxdis)
			maxdis = tmpdis;
		u += delta;
	}
	return maxdis;
}

/*计算有理曲线豪斯多夫距离: 离散方法*/
double calculateMaxDisBetweenRationalCrvLen(tinynurbs::RationalCurve<double> crv, double u1, double u2) {
	//double sum = 0;
	double A, B, C;
	glm::vec3 data1 = tinynurbs::curvePoint(crv, u1);
	glm::vec3 data2 = tinynurbs::curvePoint(crv, u2);

	A = (data1.y - data2.y) / (data1.x - data2.x);
	B = -1;
	C = data1.y - A * data1.x;

	double s_num = 10000;
	double delta = (u2 - u1) / s_num;
	double u = u1;

	double tmpdis;
	double maxdis = 0;

	std::vector<glm::dvec3> derivateData;
	double s;

	for (int i = 1; i <= s_num; i++)
	{
		glm::vec3 data3 = tinynurbs::curvePoint(crv, u + delta / 2);
		tmpdis = fabs(A * data3.x + B * data3.y + C) / sqrt(A * A + B * B);
		if (tmpdis > maxdis)
			maxdis = tmpdis;
		u += delta;
	}
	return maxdis;
}

/*计算在某一段曲线上的最大曲率: 离散方法*/
double calculateMaxCurvatureOnPartCurve(tinynurbs::Curve<double> crv, double u1, double u2) {

	//double sum = 0;
	double s_num = 10000;
	double delta = (u2 - u1) / s_num;
	double u = u1;
	double curvature;
	double maxcurvature = 0;
	std::vector<glm::dvec3> derivateData;
	double s;
	for (int i = 1; i <= s_num; i++)
	{
		curvature = calculateCurvePointCurvature(crv, u + delta / 2);
		if (curvature > maxcurvature)
			maxcurvature = curvature;

		u += delta;
	}
	return maxcurvature;
}

/*2013年论文中的评判指标*/
double calculate2013Criteria(tinynurbs::Curve<double> crv, double u1, double u2) {

	double maxdis = calculateMaxDisBetweenCrvLen(crv, u1, u2);
	double maxcur = calculateMaxCurvatureOnPartCurve(crv, u1, u2);
	return (1 + maxcur) * maxdis;
}

double calculateModifiedCriteria(tinynurbs::Curve<double> crv, double u1, double u2) {

	double maxdis = calculateMaxDisBetweenCrvLen(crv, u1, u2);
	double maxcur = calculateMaxCurvatureOnPartCurve(crv, u1, u2);
	return maxcur/(1+maxcur) * maxdis;
}

/* 节点细化*/
void DecomposeCurve(int n, int p, std::vector<double> knots, std::vector<double> Pweight, std::vector<glm::dvec3> Pw, 
					std::vector<std::vector<double>>& Qweight, int& nb, std::vector<std::vector<glm::dvec3>>& Qw){

	for (int i = 0; i < Pw.size(); i++)
	{
		Pw[i].x *= Pweight[i];
		Pw[i].y *= Pweight[i];
		Pw[i].z = Pweight[i];
	}
	int m = n + p + 1;
	int a = p;
	int b = p + 1;

	Qw.resize(n - p + 1);
	for (int i = 0; i < n - p + 1; i++)
	{
		Qw[i].resize(p + 1);
	}
	nb = 0;
	for (int i = 0; i <= p; i++)
	{
		Qw[nb][i] = Pw[i];
	}
	while (b < m) {
		int i = b;
		while (b < m && knots[b + 1] == knots[b]) b++;
		int mult = b - i + 1;
		if (mult < p) {
			double numer = knots[b] - knots[a];
			std::vector<double> alphas(p);
			for (int j = p; j > mult; j--) {
				alphas[j - mult - 1] = numer / (knots[a + j] - knots[a]);
			}
			int r = p - mult;
			for (int j = 1; j <= r; j++)
			{
				int save = r - j;
				int s = mult + j;
				for (int k = p; k >= s; k--)
				{
					double alpha = alphas[k - s];
					Qw[nb][k].x = alpha * Qw[nb][k].x + (1.0 - alpha) * Qw[nb][k - 1].x;
					Qw[nb][k].y = alpha * Qw[nb][k].y + (1.0 - alpha) * Qw[nb][k - 1].y;
					Qw[nb][k].z = alpha * Qw[nb][k].z + (1.0 - alpha) * Qw[nb][k - 1].z;

				}
				if (b < m)
					Qw[nb + 1][save] = Qw[nb][p];
			}

		}
		nb += 1;
		if (b < m) {
			for (i = p - mult; i <= p; i++)
				Qw[nb][i] = Pw[b - p + i];
			a = b;
			b = b + 1;
		}
	}

	for (int i = nb; i < n - p + 1; i++) {
		Qw.pop_back();
	}

	for (int i = 0; i < Qw.size(); i++)
	{
		std::vector<double> tmp;
		for (int j = 0; j < Qw[i].size(); j++)
		{
			Qw[i][j].x /= Qw[i][j].z;
			Qw[i][j].y /= Qw[i][j].z;
			tmp.push_back(Qw[i][j].z);
		}
		Qweight.push_back(tmp);
	}

	double rescrvlen = 0;
	double resintegral = 0;
	std::vector<double> tmpknots;
	for (int i = 0; i <= p; i++)
	{
		tmpknots.push_back(0);
	}
	for (int i = 0; i <= p; i++)
	{
		tmpknots.push_back(1);
	}
	for (int i = 0; i < nb; i++)
	{
		tinynurbs::RationalCurve<double> testcrv;
		testcrv.knots = tmpknots;
		testcrv.degree = p;
		testcrv.weights = Qweight[i];
		testcrv.control_points = Qw[i];
		double linelen, crvlen;
		calculateRationalCurveLength(testcrv, 0, 1, linelen, crvlen, 1);

		resintegral += calculateRationalCurveCurvatureIntegral(testcrv, 1);

		rescrvlen += crvlen;

	}
	printf("Compose Len: %.6f %.6f\n", rescrvlen, resintegral);

}

/*节点细化之后计算并返回曲率平方沿曲线积分和曲线总长度*/
void DecomposeCurveToSave(int n, int p, std::vector<double> knots, std::vector<double> Pweight, std::vector<glm::dvec3> Pw,
	std::vector<std::vector<double>>& Qweight, int& nb, std::vector<std::vector<glm::dvec3>>& Qw,  double& resintegral) {

	double rescrvlen;


	for (int i = 0; i < Pw.size(); i++)
	{
		Pw[i].x *= Pweight[i];
		Pw[i].y *= Pweight[i];
		Pw[i].z = Pweight[i];
	}


	int m = n + p + 1;
	int a = p;
	int b = p + 1;

	Qw.resize(n - p + 1);
	for (int i = 0; i < n - p + 1; i++)
	{
		Qw[i].resize(p + 1);
	}
	nb = 0;
	for (int i = 0; i <= p; i++)
	{
		Qw[nb][i] = Pw[i];
	}
	while (b < m) {
		int i = b;
		while (b < m && knots[b + 1] == knots[b]) b++;
		int mult = b - i + 1;
		if (mult < p) {
			double numer = knots[b] - knots[a];
			std::vector<double> alphas(p);
			for (int j = p; j > mult; j--) {
				alphas[j - mult - 1] = numer / (knots[a + j] - knots[a]);
			}
			int r = p - mult;
			for (int j = 1; j <= r; j++)
			{
				int save = r - j;
				int s = mult + j;
				for (int k = p; k >= s; k--)
				{
					double alpha = alphas[k - s];
					Qw[nb][k].x = alpha * Qw[nb][k].x + (1.0 - alpha) * Qw[nb][k - 1].x;
					Qw[nb][k].y = alpha * Qw[nb][k].y + (1.0 - alpha) * Qw[nb][k - 1].y;
					Qw[nb][k].z = alpha * Qw[nb][k].z + (1.0 - alpha) * Qw[nb][k - 1].z;

				}
				if (b < m)
					Qw[nb + 1][save] = Qw[nb][p];
			}

		}
		nb += 1;
		if (b < m) {
			for (i = p - mult; i <= p; i++)
				Qw[nb][i] = Pw[b - p + i];
			a = b;
			b = b + 1;
		}
	}

	for (int i = nb; i < n - p + 1; i++) {
		Qw.pop_back();
	}

	for (int i = 0; i < Qw.size(); i++)
	{
		std::vector<double> tmp;
		for (int j = 0; j < Qw[i].size(); j++)
		{
			Qw[i][j].x /= Qw[i][j].z;
			Qw[i][j].y /= Qw[i][j].z;
			tmp.push_back(Qw[i][j].z);
		}
		Qweight.push_back(tmp);
	}

	rescrvlen = 0;
	resintegral = 0;
	std::vector<double> tmpknots;
	for (int i = 0; i <= p; i++)
	{
		tmpknots.push_back(0);
	}
	for (int i = 0; i <= p; i++)
	{
		tmpknots.push_back(1);
	}
	for (int i = 0; i < nb; i++)
	{
		tinynurbs::Curve<double> testcrv;
		//tinynurbs::Curve<double> testcrv;
		testcrv.knots = tmpknots;
		testcrv.degree = p;
		//testcrv.weights = Qweight[i];
		testcrv.control_points = Qw[i];
		double linelen, crvlen;


		crvlen = XinPuSenCurveLength(0, 1, testcrv, 1e-5, 10000000);

		//resintegral += calculateRationalCurveCurvatureIntegral(testcrv, 1);

		resintegral += calculateCurveCurvatureIntegral(testcrv, 1);

		rescrvlen += crvlen;

	}
	//printf("Compose Len: %.6f %.6f\n", rescrvlen, resintegral);

}

/*解析计算一段三次贝塞尔上拐点个数，传入参数为控制点坐标*/
int calculateCurveTurningNum(std::vector<glm::dvec3> P) {

	double A, B, C, D, E, F, G, H, I, J, K, L;
	A = P[0].x * P[1].y - P[1].x * P[0].y;
	B = P[0].x * P[2].y - P[2].x * P[0].y;
	C = P[0].x * P[3].y - P[3].x * P[0].y;

	D = P[1].x * P[0].y - P[0].x * P[1].y;
	E = P[1].x * P[2].y - P[2].x * P[1].y;
	F = P[1].x * P[3].y - P[3].x * P[1].y;

	G = P[2].x * P[0].y - P[0].x * P[2].y;
	H = P[2].x * P[1].y - P[1].x * P[2].y;
	I = P[2].x * P[3].y - P[3].x * P[2].y;

	J = P[3].x * P[0].y - P[0].x * P[3].y;
	K = P[3].x * P[1].y - P[1].x * P[3].y;
	L = P[3].x * P[2].y - P[2].x * P[3].y;

	double a, b, c, d;

	a = -54 * A + 54 * B - 18 * C - 54 * D - 162 * E + 54 * F + 54 * G - 162 * H - 54 * I
		- 18 * J + 54 * K - 54 * L;

	b = 144 * A - 126 * B + 36 * C + 126 * D + 270 * E - 72 * F - 90 * G + 216 * H + 36 * I
		+ 18 * J - 36 * K + 18 * L;

	c = -126 * A + 90 * B - 18 * C - 90 * D - 126 * E + 18 * F + 36 * G - 72 * H;

	d = 36 * A - 18 * B + 18 * D + 18 * E;

	/*
	a = 1;
	b = -2;
	c = 1;
	d = 0;
	*/
	double p, q, delta;
	double x0, x1, x2;
	int num = 0;

	if (abs(a) > eps) {

		//std::cout << "test" << std::endl;

		p = c / a - 3 * pow(b / (3 * a), 2);
		q = d / a - (b / (3 * a)) * (c / a) + 2 * pow(b / (3 * a), 3);

		delta = pow(q / 2, 2) + pow(p / 3, 3);
		std::cout << "delta: " << delta << std::endl;
		std::cout << "p q: " << p << " " << q << std::endl;

		if (abs(delta) < eps) {
			if (abs(p - q) < eps && abs(p) < eps) {
				return 1;
			}
			else if (abs(p * q) > eps) {
				x0 = cbrt(-4 * q);
				x1 = cbrt(q / 2);

				x0 -= b / (3 * a);
				x1 -= b / (3 * a);

				std::cout << "x0 x1: " << x0 << " " << x1 << std::endl;

				if (x0 >= -eps && x0 <= 1)
					num++;
				if (x1 >= -eps && x1 <= 1)
					num++;
			}
		}
		else if (delta >= eps) {
			x0 = cbrt(-q / 2 + pow(pow(q / 2, 2) + pow(p / 3, 3), 1.0 / 3))
				+ cbrt(-q / 2 - pow(pow(q / 2, 2) + pow(p / 3, 3), 1.0 / 3));

			x0 -= b / (3 * a);
			if (x0 >= -eps && x0 <= 1)
				num++;
		}
		else {
			double tmp = acos((-q / 2) * pow(-p / 3, -1.5));
			for (int i = 0; i <= 2; i++)
			{
				x0 = 2 * sqrt(-p / 3) * cos((tmp + 2 * i * PI) / 3);
				if (x0 >= -eps && x0 <= 1) {
					num++;
				}
			}
		}

	}
	else if (abs(b) > eps) {
		double delta = c * c - 4 * b * d;
		if (delta < 0) {
			return 0;
		}
		else if (abs(delta) < eps) {
			x0 = -c / (2 * b);
			if (x0 >= -eps && x0 <= 1)
				num++;
		}
		else {
			x0 = (-c + delta) / (2 * b);
			x1 = (-c - delta) / (2 * b);
			if (x0 >= -eps && x0 <= 1)
				num++;
			if (x1 >= -eps && x1 <= 1)
				num++;
		}
	}
	else if (abs(c) > eps) {
		x0 = -d / c;
		if (x0 >= -eps && x0 <= 1)
			num++;
	}

	return num;

}

/*利用刘鼎元计算一段三次贝塞尔曲线上拐点个数，传入参数为控制点坐标*/
int calculateCurveTurningNumLiu(std::vector<glm::dvec3> P) {
	int num;
	glm::dvec3 a1, a2, a3;

	a1.x = P[1].x - P[0].x;
	a1.y = P[1].y - P[0].y;

	a2.x = P[2].x - P[1].x;
	a2.y = P[2].y - P[1].y;

	a3.x = P[3].x - P[2].x;
	a3.y = P[3].y - P[2].y;

	double A1, A2, A3;
	A1 = a2.x * a3.y - a3.x * a2.y;
	A2 = a3.x * a1.y - a1.x * a3.y;
	A3 = a1.x * a2.y - a2.x * a1.y;

	if (A1 * A3 < -eps) {
		num = 1;
	}
	else if (A1 * A3 > eps)
	{
		double a, b, c, d;
		a = A3;
		b = A2 + 2 * A3;
		c = A1 + A2 + A3;
		d = b / c;
		if (d < 2 && A2 * A2 > 4 * A1 * A3 && a * b > eps && b * c > eps) {
			num = 2;
		}
		else {
			num = 0;
		}
	}
	else {
		num = 0;
	}
	return num;
}



/*计算一段三次贝塞尔曲线跟一段折线的豪斯多夫距离
*贝塞尔曲线控制点坐标
* 曲线参数范围
* 折线两端点坐标
*/

double CubicFun(double a, double b, double c, double d, double u) {
	return  a * pow(u, 3) + b * pow(u, 2) + c * u + d;
}

double CalculateHausdorffDistanceBetweenBL(std::vector<glm::dvec3> C, double u1, double u2, glm::dvec3 p1, glm::dvec3 p2) {

	double maxdis = 0;
	Eigen::Matrix<double, 3, 3> trans, anticlockwise, clockwise, roate;

	Eigen::Matrix<double, 3, 1> mp1, mp2;

	mp1 << p1.x,
		p1.y,
		1;
	mp2 << p2.x,
		p2.y,
		1;

	trans << 1, 0, -p1.x,
		0, 1, -p1.y,
		0, 0, 1;
	mp1 = trans * mp1;
	mp2 = trans * mp2;

	double cosa = mp2(0, 0) / sqrt(mp2(0, 0) * mp2(0, 0) + mp2(1, 0) * mp2(1, 0));
	double sina = mp2(1, 0) / sqrt(mp2(0, 0) * mp2(0, 0) + mp2(1, 0) * mp2(1, 0));

	roate << cosa, sina, 0,
		-sina, cosa, 0,
		0, 0, 1;

	mp2 = roate * mp2;

	std::vector<double> C_y;

	for (int i = 0; i < C.size(); i++)
	{
		Eigen::Matrix<double, 3, 1> tmp;
		tmp << C[i].x,
			C[i].y,
			C[i].z;
		tmp = roate * trans * tmp;
		C_y.push_back(tmp(1, 0));
	}

	double a, b, c, d;
	a = -C_y[0] + 3 * C_y[1] - 3 * C_y[2] + C_y[3];
	b = 3 * C_y[0] - 6 * C_y[1] + 3 * C_y[2];
	c = -3 * C_y[0] + 3 * C_y[1];
	d = C_y[0];
	//std::cout << a << " " << b << " " << c << " " << d << std::endl;
	double res1, res2;
	res1 = fabs(CubicFun(a, b, c, d, u1));
	res2 = fabs(CubicFun(a, b, c, d, u2));
	if (res1 > maxdis) {
		maxdis = res1;
	}
	if (res2 > maxdis) {
		maxdis = res2;
	}

	double a1, b1, c1;
	a1 = 3 * a;
	b1 = 2 * b;
	c1 = c;
	if (a1 != 0) {
		double delta = b1 * b1 - 4 * a1 * c1;

		if (delta >= 0) {
			double x1 = (-b1 + delta) / (2 * a1);
			if (x1 > u1 && x1 < u2) {
				double res = fabs(CubicFun(a, b, c, d, x1));
				if (res > maxdis) {
					maxdis = res;
				}
			}

			if (delta > 0) {
				double x2 = (-b1 - delta) / (2 * a1);
				if (x2 > u1 && x2 < u2) {
					double res = fabs(CubicFun(a, b, c, d, x2));
					if (res > maxdis) {
						maxdis = res;
					}
				}
			}
		}
	}
	else if (b1 != 0) {
		double x1 = -c1 / b1;
		if (x1 > u1 && x1 < u2) {
			double res = fabs(CubicFun(a, b, c, d, x1));
			if (res > maxdis) {
				maxdis = res;
			}
		}
	}

	//std::cout << mp2(0, 0) << " " << mp2(1, 0) << std::endl;
	//std::cout << maxdis << std::endl;
	return maxdis;
}

/*计算每一段的数据点的豪斯多夫距离
* 所有贝塞尔曲线控制点
* 所有数据点
* 参数
* 节点
*/
vector<double> CalculateHausdorffDistance(vector<vector<glm::dvec3>> C, vector<glm::dvec3> P, vector<double> Param, vector<double> Knots) {

	vector<double> max_dis;

	for (int i = 0; i < P.size() - 1; i++) {

		double p1, p2;
		p1 = Param[i];
		p2 = Param[i + 1];
		vector<int> knots_id;
		int j;
		for (j = 3; Knots[j] < p2; j++) {
			if (Knots[j] == p1) {
				knots_id.push_back(j);
			}
			else if (Knots[j] > p1) {
				if (knots_id.size() == 0)
					knots_id.push_back(j - 1);
				knots_id.push_back(j);
			}

		}
		if (knots_id.size() == 0)
			knots_id.push_back(j - 1);
		knots_id.push_back(j);

		double max = 0;

		for (j = 1; j < knots_id.size(); j++)
		{
			double u1, u2, tmp;
			if (Knots[knots_id[j - 1]] <= p1) {
				u1 = (p1 - Knots[knots_id[j - 1]]) / (Knots[knots_id[j]] - Knots[knots_id[j - 1]]);
				if (Knots[knots_id[j]] <= p2) {
					u2 = 1.0;
				}
				else {
					u2 = (p2 - Knots[knots_id[j - 1]]) / (Knots[knots_id[j]] - Knots[knots_id[j - 1]]);
				}
			}
			else
			{
				u1 = 0;
				if (Knots[knots_id[j]] <= p2) {
					u2 = 1.0;
				}
				else {
					u2 = (p2 - Knots[knots_id[j - 1]]) / (Knots[knots_id[j]] - Knots[knots_id[j - 1]]);
				}
			}
			tmp = CalculateHausdorffDistanceBetweenBL(C[knots_id[j] - 3], u1, u2, P[i], P[i + 1]);
			if (tmp > max)
				max = tmp;
		}
		max_dis.push_back(max);
	}

	return max_dis;

}


long double fun(long double x) {
	long double res = sqrt(1 - x * x);
	return res;
}

long double calculatGaussLengendre(long double u1, long double u2) {



	long double x5[5] = { 0.0, -0.5384693101056831, 0.5384693101056831, -0.9061798459386640, 0.9061798459386640 };

	long double w5[5] = { 0.56888888888889, 0.4786286704993665, 0.4786286704993665, 0.2369268850561891, 0.2369268850561891 };


	long double x6[6] = { -0.9324695142, -0.6612093865, -0.2386191861,
						  0.2386191861, 0.6612093865, 0.9324695142 };
	long double w6[6] = { 0.1713244924, 0.3607615730,  0.4679139346,
							0.4679139346, 0.3607615730, 0.1713244924 };

	long double x8[8] = { -0.9602898565, -0.7966664774, -0.5255324099, -0.1834346425,
					0.1834346425, 0.5255324099, 0.7966664774, 0.9602898565 };
	long double w8[8] = { 0.1012285363, 0.2223810345, 0.3137066459, 0.3626837834,
					0.3626837834,  0.3137066459, 0.2223810345, 0.1012285363 };

	int m = 1;
	long double sum_res = 0;
	long double delta = (u2 - u1) / m;
	u2 = u1;
	for (int i = 0; i < m; i++) {
		u1 = u2;
		u2 = u1 + delta;
		long double res = 0;
		//std::vector<glm::dvec3> derivateData;
		int n = 5;
		for (int i = 0; i < n; i++) {

			long double u = x5[i] * (u2 - u1) / 2.0 + (u1 + u2) / 2.0;

			//res += w[i]*sqrt(1 + u * u / (1 - u * u));

			res += w5[i] * fun(u);

		}

		res *= (u2 - u1) / 2.0;

		sum_res += res;
	}

	return sum_res;
}

long double calculateXinpusen(long double u1, long double u2) {
	int n = 1000;
	long double res = 0;

	long double delta = (u2 - u1) / n;

	long double tmpu = u1;
	for (int i = 0; i <= n; i++) {
		if (i == 0 || i == n) {
			res += fun(tmpu);
		}
		else if (i % 2 == 0) {
			res += 2 * fun(tmpu);
		}
		else
		{
			res += 4 * fun(tmpu);
		}
		tmpu += delta;
	}
	res = res * (u2 - u1) / n / 3;

	return res;
}





