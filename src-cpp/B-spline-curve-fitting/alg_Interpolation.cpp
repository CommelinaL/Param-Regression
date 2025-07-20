#include "tinynurbs.h"
#include "alg_Interpolation.h"
#include <Eigen/Dense>
#include <iostream>
#include <math.h>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include "CalculateCurve.h"
#include "GenerateData.h"
//#define NATURAL

void closed_knots(int n, int p, const std::vector<double>& U0,
	std::vector<double>& U)
{
	assert(U0.size() == n - p + 2);
	U.resize(n + p + 2);
	for (int i = 0; i <= n - p + 1; i++)
	{
		U[i + p] = U0[i];
	}
	double d = U0[n - p + 1] - U0[0];
	for (int i = 0; i < p; i++)
	{
		U[i] = U0[i + n - p + 1 - p] - d;
		U[i + n + 2] = U0[i + 1] + d;
	}
}

void closed_lls(const std::vector<glm::dvec3>& points, int degree, std::vector<double>& knots,
	const std::vector<double>& param, std::vector<glm::dvec3>& controlPts) {
	int n = points.size();
	Eigen::MatrixXd matN = Eigen::MatrixXd::Zero(n + 1, n - degree + 1);
	Eigen::MatrixXd D_0(n, 1);
	Eigen::MatrixXd D_1(n, 1);
	Eigen::MatrixXd P_0(n, 1);
	Eigen::MatrixXd P_1(n, 1);
	for (int i = 0; i <= n; i++) {
		for (int j = 0; j < n; j++) {
			matN(j, i) = tinynurbs::bsplineOneBasis(i, degree, knots, param[j]);
		}
		D_0(i, 0) = points[i].x;
		D_1(i, 0) = points[i].y;
	}
	P_0 = matN.colPivHouseholderQr().solve(D_0);
	P_1 = matN.colPivHouseholderQr().solve(D_1);
	for (int i = 0; i < n; i++) {
		controlPts[i].x = P_0(i, 0);
		controlPts[i].y = P_1(i, 0);
		controlPts[i].z = 0;
	}
}

//void closed_lls(BSpline& l, MatrixX2d& data_points, VectorXd& parameters)
//{
//	int n = l.n, p = l.p, m = data_points.rows() - 1;
//
//	MatrixXd N = MatrixXd::Zero(m + 1, n - p + 1);
//	for (int i = 0; i <= m; i++)
//	{
//		int span = l.find_span(parameters(i));
//		VectorXd Ni;
//		l.basis(parameters(i), span, p, Ni);
//		for (int j = 0; j <= p; j++)
//		{
//			N(i, (span - p + j) % (n - p + 1)) = Ni(j);
//		}
//	}
//
//	MatrixX2d R = data_points;
//
//	MatrixX2d P = (N.transpose() * N).ldlt().solve(N.transpose() * R);
//	for (int i = 0; i <= n; i++)
//	{
//		l.P.row(i) = P.row(i % (n - p + 1));
//	}
//}

void AvgKnots(int degree, std::vector<double>& knots,
	const std::vector<double>& param) {
	// Computing knots using the average method
	int n = param.size();
	int m = n + degree + 1;

	knots.resize(m);

	double num = m - 2 * degree - 2;

	for (int i = 0; i <= degree; i++)knots[i] = 0;
	for (int i = degree + 1; i < degree + num + 1; i++) {
		double sum = 0;
		for (int k = i - degree; k <= i - 1; k++) {
			sum += param[k];
		}
		knots[i] = sum / (double)degree;
	}
	for (int i = degree + num + 1; i < m; i++) knots[i] = 1;
}

void KTPKnots(int degree, std::vector<double>& knots,
	const std::vector<double>& param){
	int n = param.size();
	int m = n + degree + 1;
	knots.resize(m);
	for (int i = 0; i <= degree; i++){
		knots[i] = 0;
		knots[m - i - 1] = 1;
	}

	double d = (double)(n) / (double)(n - degree);
	for (int j = 1; j < n - degree; j++){
		int i = floor(j * d);
		double alpha = j * d - (double)i;
		knots[degree + j] = (1 - alpha) * param[i-1] + alpha * param[i];
	}
}

void UniformKnots(int degree, std::vector<double>& knots,
	const std::vector<double>& param) {
	int n = param.size();
	int m = n + degree + 1;
	knots.resize(m);
	double num = m - 2 * degree - 2;
	for (int i = 0; i <= degree; i++)knots[i] = 0;
	for (int i = degree + 1; i < degree + num + 1; i++)knots[i] = double(i - degree) / double(num + 1);
	for (int i = degree + num + 1; i < m; i++)knots[i] = 1;
}

void NaturalKnots(int degree, std::vector<double>& knots,
	const std::vector<double>& param) {
	int n = param.size();
	int m = n + degree + 1;
	knots.resize(m);
	//domain knots
	for (int i = 0; i <= degree; i++){
	    knots[i] = 0;
	    knots[m - i - 1] = 1;
    }
	for (int i = degree + 1; i < n + degree; i++)
	{
		knots[i] = param[i - degree];
	}
}

//Solve linear equations to find the control points for the fitting curve 
void GlobalInterp(const std::vector<glm::dvec3>& points, int degree, const std::vector<double>& knots,
	const std::vector<double>& param, std::vector<glm::dvec3>& controlPts) {
	int n = points.size();
	controlPts.resize(n);

	//std::cout << "controlPts_num: " << n << std::endl;

	Eigen::MatrixXd matN(n, n);
	Eigen::MatrixXd D_0(n, 1);
	Eigen::MatrixXd D_1(n, 1);
	Eigen::MatrixXd P_0(n, 1);
	Eigen::MatrixXd P_1(n, 1);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			matN(j, i) = tinynurbs::bsplineOneBasis(i, degree, knots, param[j]);
		}
		D_0(i, 0) = points[i].x;
		D_1(i, 0) = points[i].y;
	}
	P_0 = matN.colPivHouseholderQr().solve(D_0);
	P_1 = matN.colPivHouseholderQr().solve(D_1);
	for (int i = 0; i < n; i++) {
		controlPts[i].x = P_0(i, 0);
		controlPts[i].y = P_1(i, 0);
		controlPts[i].z = 0;

	//	std::cout << "controlPts: " << controlPts[i].x <<" "<< controlPts[i].x  << std::endl;
	}

}

double conditionNumber(const Eigen::MatrixXd& matrix) {
	if (matrix.rows() != matrix.cols()) {
		throw std::invalid_argument("Matrix must be square for condition number calculation.");
	}

	Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix);
	double sigma_max = svd.singularValues()(0);        // max singular value
	double sigma_min = svd.singularValues().tail(1)(0); // min singular value

	if (sigma_min < 1e-10) { // dealing with singular or ill-conditioned matrix
		std::cerr << "Warning: Matrix is singular or ill-conditioned (min_sigma ≈ 0)." << std::endl;
		return std::numeric_limits<double>::infinity();
	}

	return sigma_max / sigma_min;
}

double paramKnotConditionNumber(int degree, const std::vector<double>& knots,
	const std::vector<double>& param) {
	int n = param.size();
	Eigen::MatrixXd matN(n, n);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			matN(j, i) = tinynurbs::bsplineOneBasis(i, degree, knots, param[j]);
		}
	}
	return conditionNumber(matN);
}

/*封闭曲线下利用平均法根据参数计算节点矢量*/
void ComputeKnots_Avg_Closed(int degree, std::vector<double>& knots,
	std::vector<double>& param) {

	int n = param.size();
	/*int m = n + degree + 1;
	knots.resize(m);
	double num = m - 2 * degree - 2;
	knots[degree] = 0;
	knots[n] = 1;
	for (int i = degree + 1; i < n; i++) {
		double sum = 0;
		for (int k = i - degree; k <= i - 1; k++) {
			sum += param[k];
		}
		knots[i] = sum / (double)degree;
	}
	for (int j = 1; j <= degree; j++) {
		knots[degree - j] = knots[degree + 1 - j] + knots[n - j] - knots[n + 1 - j];
		knots[n + j] = knots[n - 1 + j] + knots[degree + j + 1] - knots[degree + j];
	}*/
	
	int m = n + 2 * degree + 1;
	knots.resize(m);
	knots[degree] = 0;
	knots[degree + n] = 1;

	//domain knots
	int d = std::ceil((degree - 1) / 2.0);

	for (int i = degree + 1; i < n + degree; i++)
	{
		double tmp = 0;
		for (int j = i - d; j <= i + d; j++) {
			if (j < degree) {
				tmp += (-knots[degree + n] + param[n + j - degree]);
			}
			else if (j >= degree + n) {
				tmp += (knots[degree + n] + param[j - degree - n]);
			}
			else {
				tmp += param[j - degree];
			}
		}
		knots[i] = tmp / (2 * d + 1);
	}
	for (int j = 1; j <= degree; j++) {
		knots[degree - j] = knots[degree + 1 - j] + knots[n + degree - 1 - j] - knots[n + degree - j];
		knots[n +degree + j] = knots[n + degree - 1 + j] + knots[degree + j + 1] - knots[degree + j];
	}

	////前p个节点矢量
	//for (int i = degree - 1; i >= 0; i--)
	//{
	//	knots[i] = knots[i + 1] + knots[n + i] - knots[n + i + 1];
	//}

	////后p个节点矢量
	//for (int i = n + degree + 1; i < n + degree + degree + 1; i++)
	//{
	//	knots[i] = knots[i - 1] + knots[i - n] - knots[i - n - 1];
	//}
	for (double k : knots) {
		std::cout << k << " ";
	}
	std::cout << endl;
}

/*封闭曲线下利用自然法根据参数计算节点矢量*/
void ComputeKnots_Natural_Closed(int degree, std::vector<double>& knots,
	std::vector<double>& param) {

	int n = param.size();
	int m = n + 2 * degree + 1;
	knots.resize(m);
	knots[degree] = 0;
	knots[degree + n] = 1;

	//domain knots
	for (int i = degree + 1; i < n + degree; i++)
	{
		knots[i] = param[i - degree];
	}

	////前p个节点矢量
	//for (int i = degree - 1; i >= 0; i--)
	//{
	//	knots[i] = knots[i + 1] + knots[n + i] - knots[n + i + 1];
	//}

	////后p个节点矢量
	//for (int i = n + degree + 1; i < n + degree + degree + 1; i++)
	//{
	//	knots[i] = knots[i - 1] + knots[i - n] - knots[i - n - 1];
	//}
	for (int j = 1; j <= degree; j++) {
		knots[degree - j] = knots[degree + 1 - j] + knots[n + degree - 1 - j] - knots[n + degree - j];
		knots[n + degree + j] = knots[n + degree - 1 + j] + knots[degree + j + 1] - knots[degree + j];
	}
	for (double k : knots) {
		std::cout << k << " ";
	}
	std::cout << endl;
}

void InterpClosed(std::vector<glm::dvec3> points, int degree, std::vector<double>& knots,
	std::vector<double>& param, std::vector<glm::dvec3>& controlPts) {
	int n = points.size();

	controlPts.resize(n + degree);

	//std::cout << "controlPts_num: " << n << std::endl;

	Eigen::MatrixXd matN(n + degree, n + degree);
	Eigen::MatrixXd D_0(n + degree, 1);
	Eigen::MatrixXd D_1(n + degree, 1);
	Eigen::MatrixXd P_0(n + degree, 1);
	Eigen::MatrixXd P_1(n + degree, 1);
	for (int i = 0; i < n + degree; i++) {
		if (i < n) {
			for (int j = 0; j < n + degree; j++) {
				matN(i, j) = tinynurbs::bsplineOneBasis(j, degree, knots, param[i]);
			}
			D_0(i, 0) = points[i].x;
			D_1(i, 0) = points[i].y;
		}
		else {
			for (int j = 0; j < n + degree; j++) {
				if (j == i - n) {
					matN(i, j) = 1;
				}
				else if (j == i) {
					matN(i, j) = -1;
				}
				else {
					matN(i, j) = 0;
				}
			}
			D_0(i, 0) = 0;
			D_1(i, 0) = 0;
		}
	}
	P_0 = matN.colPivHouseholderQr().solve(D_0);
	P_1 = matN.colPivHouseholderQr().solve(D_1);
	for (int i = 0; i < n + degree; i++) {
		controlPts[i].x = P_0(i, 0);
		controlPts[i].y = P_1(i, 0);
		controlPts[i].z = 0;
		std::cout << "controlPts: " << controlPts[i].x << " " << controlPts[i].y << std::endl;
	}
}

/*函数功能：反求封闭曲线的控制点
*p次的封闭曲线的前p个控制点和后p个控制点重合
*/
void GlobalInterp_Closed(std::vector<glm::dvec3> points, int degree, std::vector<double>& knots,
	std::vector<double>& param, std::vector<glm::dvec3>& controlPts) {
#ifdef NATURAL
	ComputeKnots_Natural_Closed(degree, knots, param);
#else
	ComputeKnots_Avg_Closed(degree, knots, param);
#endif
	InterpClosed(points, degree, knots, param, controlPts);
}

//Use the uniform method to set the parameters and the knot vectors
void UniformInterp(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param) {

	int n = points.size();
	int m = n + degree + 1;

	param.resize(n);

	for (int i = 0; i < n; i++) {
		param[i] = (double)i / (double)(n - 1);
	}



	/*
	double num = m - 2 * degree - 2;
	double delta = (double)1.0 / num;
	for (int i = 0; i <= degree; i++)knots[i] = 0;
	for (int i = degree + 1; i < degree + num + 1; i++)knots[i] = double(i - degree) / double(num + 1);
	for (int i = degree + num + 1; i < m; i++)knots[i] = 1;
	*/
#ifdef KTP
    KTPKnots(degree, knots, param);
#elif defined(UNIFORM_KNOTS)
	UniformKnots(degree, knots, param);
#elif defined(NATURAL)
	NaturalKnots(degree, knots, param);
#else
	AvgKnots(degree, knots, param);
#endif

	
	/*knots.resize(m);

	for (int i = 0; i <= degree; i++)knots[i] = 0;
	for (int i = degree + 1; i < degree + num + 1; i++) {
		double sum = 0;
		for (int k = i - degree; k <= i - 1; k++) {
			sum += param[k];
		}
		knots[i] = sum / (double)degree;
	}
	for (int i = degree + num + 1; i < m; i++)knots[i] = 1;*/

}

void ChordInterp(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param) {

	int n = points.size();
	int m = n + degree + 1;
	param.resize(n);

	std::vector<double> chorldLen;
	double sum = 0;

	//std::cout << "\n";
	for (int i = 1; i < n; i++) {
		double len = std::sqrt((points[i].x - points[i - 1].x) * (points[i].x - points[i - 1].x) + (points[i].y - points[i - 1].y) * (points[i].y - points[i - 1].y));

		//std::cout << len << " ";

		sum += len;
		chorldLen.push_back(sum);
	}
	//std::cout << "\n";

	param[0] = 0, param[n - 1] = 1;
	for (int i = 1; i < n - 1; i++) {
		param[i] = chorldLen[i - 1] / sum;
	}

#ifdef KTP
	KTPKnots(degree, knots, param);
#elif defined(UNIFORM_KNOTS)
	UniformKnots(degree, knots, param);
#elif defined(NATURAL)
	NaturalKnots(degree, knots, param);
#else
	AvgKnots(degree, knots, param);
#endif
	/*double num = m - 2 * degree - 2;
	knots.resize(m);
	for (int i = 0; i <= degree; i++)knots[i] = 0;
	for (int i = degree + 1; i < degree + num + 1; i++) {
		double sum = 0;
		for (int k = i - degree; k <= i - 1; k++) {
			sum += param[k];
		}
		knots[i] = sum / (double)degree;
	}
	for (int i = degree + num + 1; i < m; i++)knots[i] = 1;*/

}

void CentripetalInterp(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param) {

	int n = points.size();
	int m = n + degree + 1;
	param.resize(n);

	std::vector<double> chorldLen;
	double sum = 0;
	for (int i = 1; i < n; i++) {
		double len = std::sqrt((points[i].x - points[i - 1].x) * (points[i].x - points[i - 1].x) + (points[i].y - points[i - 1].y) * (points[i].y - points[i - 1].y));
		len = std::sqrt(len);
		sum += len;
		chorldLen.push_back(sum);
	}
	param[0] = 0, param[n - 1] = 1;
	for (int i = 1; i < n - 1; i++) {
		param[i] = chorldLen[i - 1] / sum;
	}

#ifdef KTP
	KTPKnots(degree, knots, param);
#elif defined(UNIFORM_KNOTS)
	UniformKnots(degree, knots, param);
#elif defined(NATURAL)
	NaturalKnots(degree, knots, param);
#else
	AvgKnots(degree, knots, param);
#endif
	/*double num = m - 2 * degree - 2;
	knots.resize(m);
	for (int i = 0; i <= degree; i++)knots[i] = 0;
	for (int i = degree + 1; i < degree + num + 1; i++) {
		double sum = 0;
		for (int k = i - degree; k <= i - 1; k++) {
			sum += param[k];
		}
		knots[i] = sum / (double)degree;
	}
	for (int i = degree + num + 1; i < m; i++)knots[i] = 1;*/

}

//Use the universal method to set the parameters and the knot vectors
void UniversalInterp(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param) {
	int n = points.size();
	int m = n + degree + 1;

	knots.resize(m);
	param.resize(n);
	double num = m - 2 * degree - 2;
	double delta = (double)1.0 / num;
	for (int i = 0; i <= degree; i++)knots[i] = 0;
	for (int i = degree + 1; i < degree + num + 1; i++)knots[i] = double(i - degree) / double(num + 1);
	for (int i = degree + num + 1; i < m; i++)knots[i] = 1;


	for (int i = 0; i < n; i++) {
		double tMax = 0, valMax = -1;
		for (double t = 0; t <= 1; t += 1 / 1000.0) {
			int span = tinynurbs::findSpan(degree, knots, t);
			double v = tinynurbs::bsplineOneBasis(i, degree, knots, t);
			if (v > valMax) {
				valMax = v; tMax = t;
			}
		}
		param[i] = tMax;
	}
	param[n - 1] = 1;

}

void CorrectChordInterp(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param) {

	int n = points.size();
	int m = n + degree + 1;
	param.resize(n);

	std::vector<double>  chordLength;

	std::vector<double> t;

	std::vector<double> angle;

	//double sum = 0;

	chordLength.resize(n+1);

	angle.resize(n);

	t.resize(n);

	for (int i = 1; i < n; i++)
	{
		chordLength[i] = std::sqrt((points[i].x - points[i - 1].x) * (points[i].x - points[i - 1].x) + (points[i].y - points[i - 1].y) * (points[i].y - points[i - 1].y));

		if (i < n - 1) {
			glm::vec2 a, b;
			a.x = points[i - 1].x - points[i].x;
			a.y = points[i - 1].y - points[i].y;
			b.x = points[i + 1].x - points[i].x;
			b.y = points[i + 1].y - points[i].y;

			angle[i] = acos((a.x * b.x + a.y * b.y) / (sqrt(a.x * a.x + a.y * a.y) * sqrt(b.x * b.x + b.y * b.y)));
			if (PI - angle[i] < PI / 2) {
				angle[i] = PI - angle[i];
			}else
			{
				angle[i] = PI / 2;
			}
		}
		
	}
	chordLength[0] = 0;
	chordLength[n] = 0;
	
	angle[0] = 0 ;
	angle[n-1] = 0;

	for (int i = 0; i < n; i++)
	{
		if (i == 0) {
			t[i] = 0;
		}else {
			t[i] = t[i - 1] + chordLength[i] * (1 + 3.0 / 2.0 * ((chordLength[i - 1] * angle[i - 1]) / (chordLength[i - 1] + chordLength[i]) + (chordLength[i + 1] * angle[i]) / (chordLength[i] + chordLength[i + 1])));
		}

	}

	param[0] = 0, param[n - 1] = 1;

	for (int i = 1; i < n-1; i++)
	{
		param[i] = t[i] / t[n - 1];
	}

	/*double num = m - 2 * degree - 2;
	knots.resize(m);
	for (int i = 0; i <= degree; i++)knots[i] = 0;
	for (int i = degree + 1; i < degree + num + 1; i++) {
		double sum = 0;
		for (int k = i - degree; k <= i - 1; k++) {
			sum += param[k];
		}
		knots[i] = sum / (double)degree;
	}
	for (int i = degree + num + 1; i < m; i++)knots[i] = 1;*/
#ifdef KTP
	KTPKnots(degree, knots, param);
#elif defined(UNIFORM_KNOTS)
	UniformKnots(degree, knots, param);
#elif defined(NATURAL)
	NaturalKnots(degree, knots, param);
#else
	AvgKnots(degree, knots, param);
#endif

}


void RefinedCentripetalInterp(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param) {    

		int n = points.size();
		int m = n + degree + 1;
		param.resize(n);

		std::vector<double> chorldLen;
		std::vector<double> delta;

		double sum = 0;
		double a = 0.1;

		std::vector<double> min_len;

		std::vector<double> angles;
		std::vector<double> radius;

		double max_min_len = 0;

		double avg_min_len = 0;

		min_len.push_back(0);
		angles.push_back(0);
		radius.push_back(0);
		for (int i = 1; i < n-1; i++)
		{
			glm::dvec2 l1 = glm::dvec2(points[i].x - points[i-1].x, points[i].y - points[i-1].y);
			glm::dvec2 l2 = glm::dvec2(points[i+1].x - points[i].x, points[i+1].y - points[i].y);
			double l1_len = std::sqrt(std::pow(l1.x, 2) + std::pow(l1.y, 2));
			double l2_len = std::sqrt(std::pow(l2.x, 2) + std::pow(l2.y, 2));

			if (l1_len * l2_len != 0) {

				if (fabs(l1.x - l2.x) < 1e-4 && fabs(l1.y - l2.y) < 1e-4 ) {
					angles.push_back(0);
				}
				else
				{
					angles.push_back(std::acos((l1.x * l2.x + l1.y * l2.y) / (l1_len * l2_len)));
				}	

			}
			else
			{
				angles.push_back(0);
			}

			double minlen = std::min(l1_len, l2_len);

			minlen = std::min(minlen, std::sqrt(pow((points[i + 1].x - points[i - 1].x), 2) + pow((points[i + 1].y - points[i - 1].y), 2)));

			if (angles[i] != 0) {
				radius.push_back(minlen / (2 * std::sin(angles[i] / 2)));  //angleΪ0��Ҫ�������۵�
			}
			else {
				radius.push_back(0);
			}

			min_len.push_back(minlen);

			if (max_min_len < minlen)
				max_min_len = minlen;

			avg_min_len += minlen;

		}

		avg_min_len /= (n - 2);

		angles.push_back(0);
		radius.push_back(0);

		min_len.push_back(0);

		/*
		for (int i = 1; i < n-1; i++)
		{
			min_len[i] /= max_min_len;

			//min_len[i] /= avg_min_len;

		}
		*/



		for (int i = 1; i < n; i++) {
			double len = std::sqrt((points[i].x - points[i - 1].x) * (points[i].x - points[i - 1].x) + (points[i].y - points[i - 1].y) * (points[i].y - points[i - 1]).y);
			len = std::sqrt(len);
			
			//double e = a * radius[i] * angles[i] * min_len[i] + a * radius[i - 1] * angles[i - 1] * min_len[i];

			double e = a * (radius[i] * angles[i]  + radius[i - 1] * angles[i - 1]);

			sum += (len + e);

			delta.push_back(sum);
		}

		param[0] = 0, param[n - 1] = 1;

		for (int i = 1; i < n - 1; i++) {
			param[i] = delta[i - 1] / sum;
		}

		/*double num = m - 2 * degree - 2;
		knots.resize(m);
		for (int i = 0; i <= degree; i++)knots[i] = 0;
		for (int i = degree + 1; i < degree + num + 1; i++) {
			double sum = 0;
			for (int k = i - degree; k <= i - 1; k++) {
				sum += param[k];
			}
			knots[i] = sum / (double)degree;

		}
		for (int i = degree + num + 1; i < m; i++)knots[i] = 1;*/
#ifdef KTP
		KTPKnots(degree, knots, param);
#elif defined(UNIFORM_KNOTS)
		UniformKnots(degree, knots, param);
#elif defined(NATURAL)
		NaturalKnots(degree, knots, param);
#else
		AvgKnots(degree, knots, param);
#endif
}

int cnt = 1;
double sum = 0;
void ClassfierInterp(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param ,const std::vector<int>& minis) {
	
	int n = points.size();
	int m = n + degree + 1;
	param.resize(n);

	std::vector<std::vector<double>> LocalParam; //����ÿһ�εĲ���������
	LocalParam.resize(points.size() - 3);

	std::vector<glm::dvec3> tmpPoints;
	tmpPoints.resize(4);

	std::vector<glm::dvec3> pts; //Ϊ������ͼ��㷽��
	std::vector<std::vector<double>>  Knots; //�ڵ�
	std::vector<std::vector<double>> Param; //����

	for (int k = 0; k < LocalParam.size(); k++)
	{

		/*
		std::vector<std::vector<double>>  Knots; //�ڵ�
		std::vector<std::vector<double>> Param; //����
		Knots.resize(3);
		Param.resize(3);
		*/

		std::vector<double> Knots;
		std::vector<double> Param;

		tmpPoints[0] = points[k];
		tmpPoints[1] = points[k+1];
		tmpPoints[2] = points[k+2];
		tmpPoints[3] = points[k+3];

		int min_i = minis[k];
	    //auto start = std::chrono::high_resolution_clock::now();
		if (min_i == 0) {
			/*���� Universal �Ĳ�ֵ */
			UniversalInterp(tmpPoints, 3, Knots, Param);
		}else if (min_i == 1){
			/*���� CorrectChord �Ĳ�ֵ */
			CorrectChordInterp(tmpPoints, 3, Knots, Param);
		}else {
			/*���� RefinedCentripetal �Ĳ�ֵ*/
			RefinedCentripetalInterp(tmpPoints, 3, Knots, Param);
		}

		/*auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - start);
		sum += duration.count();*/
		
		for (int i = 0; i < Param.size(); i++)
		{
			LocalParam[k].push_back(Param[i]);
		}

		//std::cout << min_i << " ";
	
	}
	/*cnt++;
	if (cnt % 400 == 0) {
		cout << "local parameter computation time:" << sum / (double)cnt << endl;
		ofstream time_record("time.txt");
		time_record << "local parameter computation time:" << sum / (double)cnt << endl;
		time_record.close();
	}*/
	std::vector<double> InteralPram;  //����ÿһ���м��ľֲ�����

	//std::cout << n - 3 << "\n";

	for (int i = 0; i < n-2; i++)
	{
		//std::cout << i << "\n";

		if (i == 0) {
			InteralPram.push_back(LocalParam[i][1] / (LocalParam[i][2] + std::numeric_limits<double>::epsilon()));
		}
		else if (i == n - 3) {
			InteralPram.push_back((LocalParam[i-1][2] - LocalParam[i-1][1]) / (LocalParam[i-1][3] - LocalParam[i-1][1] + std::numeric_limits<double>::epsilon()));
		}else{
			double s1, s2;
			s1 = (LocalParam[i-1][2] - LocalParam[i-1][1]) / (LocalParam[i-1][3] - LocalParam[i-1][1] + std::numeric_limits<double>::epsilon());
			s2 = LocalParam[i][1] / (LocalParam[i][2] + std::numeric_limits<double>::epsilon());

			InteralPram.push_back((s1 + s2) / 2);
		}

		//std::cout << InteralPram[i] << " ";

	}

	std::vector<double> f, h;

	//h: 2 - n-1
	//f: 3 - n-1
	for (int i = 0; i < InteralPram.size(); i++)
	{
		h.push_back(InteralPram[i] * (1 - InteralPram[i]));
		
		if(i>0)
			f.push_back(InteralPram[i - 1] * InteralPram[i - 1] + (1 - InteralPram[i]) * (1 - InteralPram[i]));
	}


	Eigen::MatrixXd mat(n - 3, n - 3);

	Eigen::VectorXd res(n - 3);

	Eigen::VectorXd x(n - 3);


	for (int i = 0; i < mat.rows(); i++)
	{
		if (i == 0) {
			mat(i, 0) = f[0];
			mat(i, 1) = -h[1];
			for (int j = 2; j < mat.cols(); j++) {
				mat(i, j) = 0;
			}
		}
		else if(i == n-4)
		{
			for (int j = 0; j < mat.cols() - 2; j++) {
				mat(i, j) = 0;
			}
			mat(i, mat.cols() - 2) = -h[h.size()-2];
			mat(i, mat.cols() - 1) = f[f.size()-1];
		}
		else
		{
			for (int j = 0; j < i-1; j++) {
				mat(i, j) = 0;
			}
			
			mat(i, i-1) = -h[i];
			mat(i, i) = f[i];
			mat(i, i+1) = -h[i+1];

			for (int j = i+2; j < mat.cols(); j++) {
				mat(i, j) = 0;
			}
		}
	}

	double delta2, deltan;
	delta2 = 1;
	deltan = 1;

	for (int i = 0; i < res.size(); i++)
	{
		if (i == 0) {
			res(i) = h[0] * delta2;
		}
		else if(i==res.size()-1)
		{
			res(i) = h[h.size() - 1] * deltan;
			 
		}
		else
		{
			res(i) = 0;
		}

	}

	x = mat.colPivHouseholderQr().solve(res);

	vector<double> delta;
	delta.resize(n - 1);
	double sum = 0;
	for ( int i = 0; i < delta.size(); i++)
	{
		if (i == 0)
			delta[i] = delta2;
		else if(i==delta.size()-1)
		{
			delta[i] = deltan;
		}
		else {
			delta[i] = x[i - 1];
		}
		sum += delta[i];

		delta[i] = sum;

	}

	param[0] = 0, param[n - 1] = 1;

	for (int i = 1; i < n - 1; i++) {
		param[i] = delta[i - 1] / sum;
	}

	/*double num = m - 2 * degree - 2;
	knots.resize(m);
	for (int i = 0; i <= degree; i++)knots[i] = 0;
	for (int i = degree + 1; i < degree + num + 1; i++) {
		double sum = 0;
		for (int k = i - degree; k <= i - 1; k++) {
			sum += param[k];
		}
		knots[i] = sum / (double)degree;
	}
	for (int i = degree + num + 1; i < m; i++)knots[i] = 1;*/
#ifdef KTP
	KTPKnots(degree, knots, param);
#elif defined(UNIFORM_KNOTS)
	UniformKnots(degree, knots, param);
#elif defined(NATURAL)
	NaturalKnots(degree, knots, param);
#else
	AvgKnots(degree, knots, param);
#endif
}


void RegressorInterp(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param, std::vector<double> intervals) {
	int n = points.size();
	int m = n + degree + 1;
	param.resize(n);

	std::vector<std::vector<double>> LocalInt;
	LocalInt.resize(n - 3);
	for (int i = 0; i < n - 3; i++) {
		LocalInt[i].assign(intervals.begin() + i * 3, intervals.begin() + (i + 1) * 3);
	}

	std::vector<glm::dvec3> pts;
	std::vector<std::vector<double>>  Knots;
	std::vector<std::vector<double>> Param;

	std::vector<double> InteralPram;

	for (int i = 0; i < n - 2; i++) {
		if (i == 0) {
			InteralPram.push_back(LocalInt[i][0] / (LocalInt[i][0] + LocalInt[i][1]));
		}
		else if (i == n - 3) {
			InteralPram.push_back(LocalInt[i - 1][1] / (LocalInt[i - 1][2] + LocalInt[i - 1][1]));
		}
		else {
			double s1, s2;
			s1 = LocalInt[i - 1][1] / (LocalInt[i - 1][2] + LocalInt[i - 1][1]);
			s2 = LocalInt[i][0] / (LocalInt[i][1] + LocalInt[i][0]);

			InteralPram.push_back((s1 + s2) / 2);
		}
	}

	std::vector<double> f, h;

	//h: 2 - n-1
	//f: 3 - n-1
	for (int i = 0; i < InteralPram.size(); i++)
	{
		h.push_back(InteralPram[i] * (1 - InteralPram[i]));

		if (i > 0)
			f.push_back(InteralPram[i - 1] * InteralPram[i - 1] + (1 - InteralPram[i]) * (1 - InteralPram[i]));
	}


	Eigen::MatrixXd mat(n - 3, n - 3);

	Eigen::VectorXd res(n - 3);

	Eigen::VectorXd x(n - 3);


	for (int i = 0; i < mat.rows(); i++)
	{
		if (i == 0) {
			mat(i, 0) = f[0];
			mat(i, 1) = -h[1];
			for (int j = 2; j < mat.cols(); j++) {
				mat(i, j) = 0;
			}
		}
		else if (i == n - 4)
		{
			for (int j = 0; j < mat.cols() - 2; j++) {
				mat(i, j) = 0;
			}
			mat(i, mat.cols() - 2) = -h[h.size() - 2];
			mat(i, mat.cols() - 1) = f[f.size() - 1];
		}
		else
		{
			for (int j = 0; j < i - 1; j++) {
				mat(i, j) = 0;
			}

			mat(i, i - 1) = -h[i];
			mat(i, i) = f[i];
			mat(i, i + 1) = -h[i + 1];

			for (int j = i + 2; j < mat.cols(); j++) {
				mat(i, j) = 0;
			}
		}
	}

	double delta2, deltan;
	delta2 = 1;
	deltan = 1;

	for (int i = 0; i < res.size(); i++)
	{
		if (i == 0) {
			res(i) = h[0] * delta2;
		}
		else if (i == res.size() - 1)
		{
			res(i) = h[h.size() - 1] * deltan;

		}
		else
		{
			res(i) = 0;
		}

	}

	x = mat.colPivHouseholderQr().solve(res);

	vector<double> delta;
	delta.resize(n - 1);
	double sum = 0;
	for (int i = 0; i < delta.size(); i++)
	{
		if (i == 0)
			delta[i] = delta2;
		else if (i == delta.size() - 1)
		{
			delta[i] = deltan;
		}
		else {
			delta[i] = x[i - 1];
		}
		sum += delta[i];

		delta[i] = sum;

	}

	param[0] = 0, param[n - 1] = 1;

	for (int i = 1; i < n - 1; i++) {
		param[i] = delta[i - 1] / sum;
	}
#ifdef KTP
	KTPKnots(degree, knots, param);
#elif defined(UNIFORM_KNOTS)
	UniformKnots(degree, knots, param);
#elif defined(NATURAL)
	NaturalKnots(degree, knots, param);
#else
	AvgKnots(degree, knots, param);
#endif
	/*double num = m - 2 * degree - 2;
	knots.resize(m);
	for (int i = 0; i <= degree; i++)knots[i] = 0;
	for (int i = degree + 1; i < degree + num + 1; i++) {
		double sum = 0;
		for (int k = i - degree; k <= i - 1; k++) {
			sum += param[k];
		}
		knots[i] = sum / (double)degree;
	}
	for (int i = degree + num + 1; i < m; i++)knots[i] = 1;*/
}

void VarRegressorInterp(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param, std::vector<double> intervals, int local_len) {
	int n = points.size();
	int m = n + degree + 1;
	param.resize(n);

	std::vector<std::vector<double>> LocalInt;
	LocalInt.resize(n - local_len + 1);
	for (int i = 0; i < n - local_len + 1; i++) {
		LocalInt[i].assign(intervals.begin() + i * (local_len - 1), intervals.begin() + (i + 1) * (local_len - 1));
	}

	std::vector<glm::dvec3> pts;
	std::vector<std::vector<double>>  Knots;
	std::vector<std::vector<double>> Param;

	std::vector<double> InteralPram;

	for (int i = 0; i < n - 2; i++) {
		double sum = 0;
		int sup = std::min(i, n - local_len);
		int inf = std::max(0, i - local_len + 3);
		int local_num = sup - inf + 1;
		for (int j = inf; j <= sup; j++) {
			int k = i - j;
			sum += LocalInt[j][k] / (LocalInt[j][k + 1] + LocalInt[j][k]);
		}
		InteralPram.push_back(sum / local_num);
	}

	std::vector<double> f, h;

	//h: 2 - n-1
	//f: 3 - n-1
	for (int i = 0; i < InteralPram.size(); i++)
	{
		h.push_back(InteralPram[i] * (1 - InteralPram[i]));

		if (i > 0)
			f.push_back(InteralPram[i - 1] * InteralPram[i - 1] + (1 - InteralPram[i]) * (1 - InteralPram[i]));
	}


	Eigen::MatrixXd mat(n - 3, n - 3);

	Eigen::VectorXd res(n - 3);

	Eigen::VectorXd x(n - 3);


	for (int i = 0; i < mat.rows(); i++)
	{
		if (i == 0) {
			mat(i, 0) = f[0];
			mat(i, 1) = -h[1];
			for (int j = 2; j < mat.cols(); j++) {
				mat(i, j) = 0;
			}
		}
		else if (i == n - 4)
		{
			for (int j = 0; j < mat.cols() - 2; j++) {
				mat(i, j) = 0;
			}
			mat(i, mat.cols() - 2) = -h[h.size() - 2];
			mat(i, mat.cols() - 1) = f[f.size() - 1];
		}
		else
		{
			for (int j = 0; j < i - 1; j++) {
				mat(i, j) = 0;
			}

			mat(i, i - 1) = -h[i];
			mat(i, i) = f[i];
			mat(i, i + 1) = -h[i + 1];

			for (int j = i + 2; j < mat.cols(); j++) {
				mat(i, j) = 0;
			}
		}
	}

	double delta2, deltan;
	delta2 = 1;
	deltan = 1;

	for (int i = 0; i < res.size(); i++)
	{
		if (i == 0) {
			res(i) = h[0] * delta2;
		}
		else if (i == res.size() - 1)
		{
			res(i) = h[h.size() - 1] * deltan;

		}
		else
		{
			res(i) = 0;
		}

	}

	x = mat.colPivHouseholderQr().solve(res);

	vector<double> delta;
	delta.resize(n - 1);
	double sum = 0;
	for (int i = 0; i < delta.size(); i++)
	{
		if (i == 0)
			delta[i] = delta2;
		else if (i == delta.size() - 1)
		{
			delta[i] = deltan;
		}
		else {
			delta[i] = x[i - 1];
		}
		sum += delta[i];

		delta[i] = sum;

	}

	param[0] = 0, param[n - 1] = 1;

	for (int i = 1; i < n - 1; i++) {
		param[i] = delta[i - 1] / sum;
	}
#ifdef KTP
	KTPKnots(degree, knots, param);
#elif defined(UNIFORM_KNOTS)
	UniformKnots(degree, knots, param);
#elif defined(NATURAL)
	NaturalKnots(degree, knots, param);
#else
	AvgKnots(degree, knots, param);
#endif
	/*knots.resize(m);
	double num = m - 2 * degree - 2;

	for (int i = 0; i <= degree; i++)knots[i] = 0;
	for (int i = degree + 1; i < degree + num + 1; i++) {
		double sum = 0;
		for (int k = i - degree; k <= i - 1; k++) {
			sum += param[k];
		}
		knots[i] = sum / (double)degree;
	}
	for (int i = degree + num + 1; i < m; i++)knots[i] = 1;*/
}

void ClassfierInterp1(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param, std::vector<int> minis) {

	int n = points.size();
	int m = n + degree + 1;
	knots.resize(m);
	param.resize(n);

	std::vector<std::vector<double>> LocalParam; //����ÿһ�εĲ���������
	LocalParam.resize(points.size() - 3);

	std::vector<glm::dvec3> tmpPoints;
	tmpPoints.resize(4);

	std::vector<glm::dvec3> pts; //Ϊ������ͼ��㷽��
	std::vector<std::vector<double>>  Knots; //�ڵ�
	std::vector<std::vector<double>> Param; //����

	for (int k = 0; k < LocalParam.size(); k++)
	{

		/*
		std::vector<std::vector<double>>  Knots; //�ڵ�
		std::vector<std::vector<double>> Param; //����
		Knots.resize(3);
		Param.resize(3);
		*/

		std::vector<double> Knots;
		std::vector<double> Param;

		tmpPoints[0] = points[k];
		tmpPoints[1] = points[k + 1];
		tmpPoints[2] = points[k + 2];
		tmpPoints[3] = points[k + 3];

		int min_i = minis[k];
		if (min_i == 0) {
			/*���� Universal �Ĳ�ֵ */
			UniversalInterp(tmpPoints, 3, Knots, Param);
		}
		else if (min_i == 1) {
			/*���� CorrectChord �Ĳ�ֵ */
			CorrectChordInterp(tmpPoints, 3, Knots, Param);
		}
		else {
			/*���� RefinedCentripetal �Ĳ�ֵ*/
			RefinedCentripetalInterp(tmpPoints, 3, Knots, Param);
		}

		for (int i = 0; i < Param.size(); i++)
		{
			LocalParam[k].push_back(Param[i]);
		}

		//std::cout << min_i << " ";

	}

	std::vector<double> InteralPram;  //����ÿһ���м��ľֲ�����

	//std::cout << n - 3 << "\n";

	for (int i = 0; i < n - 2; i++)
	{
		//std::cout << i << "\n";

		if (i == 0) {
			InteralPram.push_back(LocalParam[i][1] / LocalParam[i][2]);
		}
		else if (i == n - 3) {
			InteralPram.push_back((LocalParam[i - 1][2] - LocalParam[i - 1][1]) / (LocalParam[i - 1][3] - LocalParam[i - 1][1]));
		}
		else {
			double s1, s2;
			s1 = (LocalParam[i - 1][2] - LocalParam[i - 1][1]) / (LocalParam[i - 1][3] - LocalParam[i - 1][1]);
			s2 = LocalParam[i][1] / LocalParam[i][2];

			InteralPram.push_back((s1 + s2) / 2);
		}

		//std::cout << InteralPram[i] << " ";

	}

	std::vector<double> f, h, g;

	//h: 2 - n-1
	//f: 3 - n-1
	for (int i = 0; i < InteralPram.size(); i++)
	{
		f.push_back(InteralPram[i] * InteralPram[i]);
		g.push_back(InteralPram[i] * (1 - InteralPram[i]));
		h.push_back((1 - InteralPram[i]) * (1 - InteralPram[i]));

	}


	Eigen::MatrixXd mat(n - 1, n - 1);

	Eigen::VectorXd res(n - 1);

	Eigen::VectorXd x(n - 1);


	for (int i = 0; i < mat.rows(); i++)
	{
		if (i == 0) {
			mat(i, 0) = f[0];
			mat(i, 1) = -g[0];
			for (int j = 2; j < mat.cols(); j++) {
				mat(i, j) = 0;
			}
		}
		else if (i == n - 2)
		{
			for (int j = 0; j < mat.cols() - 2; j++) {
				mat(i, j) = 0;
			}
			mat(i, mat.cols() - 2) = -g[g.size() - 1];
			mat(i, mat.cols() - 1) = f[f.size() - 1];
		}
		else
		{
			for (int j = 0; j < i - 1; j++) {
				mat(i, j) = 0;
			}

			mat(i, i - 1) = -g[i-1];
			mat(i, i) = f[i-1] + h[i];
			mat(i, i + 1) = -g[i];

			for (int j = i + 2; j < mat.cols(); j++) {
				mat(i, j) = 0;
			}
		}
	}

	//double delta2, deltan;
	//delta2 = 1;
	//deltan = 1;

	for (int i = 0; i < res.size(); i++)
	{
		res(i) = 0;
	}


	Eigen::FullPivLU<Eigen::Matrix3d> lu_decomp(mat); // ʹ��LU�ֽ������ռ�  

	Eigen::Vector3d solution = lu_decomp.kernel(); // �����ռ�Ļ���

	// ��鲢��������  
	for (int i = 0; i < solution.size(); ++i) {
		if (solution(i) != 0) {
			std::cout << "Non-zero solution found: " << solution.transpose() << std::endl;
			break;
		}
	}

	x = mat.colPivHouseholderQr().solve(res);

	double sum = 0;
	std::cout << "\n";
	for (int i = 0; i < x.size(); i++)
	{
		sum += x[i];
		x[i] = sum;
		std::cout << x[i] << " ";
	}
	std::cout << "\n";

	param[0] = 0, param[n - 1] = 1;

	for (int i = 1; i < n - 1; i++) {
		param[i] = x[i - 1] / sum;
	}

	double num = m - 2 * degree - 2;

	for (int i = 0; i <= degree; i++)knots[i] = 0;
	for (int i = degree + 1; i < degree + num + 1; i++) {
		double sum = 0;
		for (int k = i - degree; k <= i - 1; k++) {
			sum += param[k];
		}
		knots[i] = sum / (double)degree;

		//std::cout << knots[i] << std::endl;
	}
	for (int i = degree + num + 1; i < m; i++)knots[i] = 1;

}




void ClassfierInterp2(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param, std::vector<int> minis) {

	int n = points.size();
	int m = n + degree + 1;
	knots.resize(m);
	param.resize(n);

	std::vector<std::vector<double>> LocalParam; //����ÿһ�εĲ���������
	LocalParam.resize(points.size() - 3);

	std::vector<glm::dvec3> tmpPoints;
	tmpPoints.resize(4);

	std::vector<glm::dvec3> pts; //Ϊ������ͼ��㷽��
	std::vector<std::vector<double>>  Knots; //�ڵ�
	std::vector<std::vector<double>> Param; //����

	for (int k = 0; k < LocalParam.size(); k++)
	{

		/*
		std::vector<std::vector<double>>  Knots; //�ڵ�
		std::vector<std::vector<double>> Param; //����
		Knots.resize(3);
		Param.resize(3);
		*/

		std::vector<double> Knots;
		std::vector<double> Param;

		tmpPoints[0] = points[k];
		tmpPoints[1] = points[k + 1];
		tmpPoints[2] = points[k + 2];
		tmpPoints[3] = points[k + 3];

		int min_i = minis[k] - 4;
		if (min_i == 0) {
			/*���� Universal �Ĳ�ֵ */
			UniversalInterp(tmpPoints, 3, Knots, Param);
		}
		else if (min_i == 1) {
			/*���� CorrectChord �Ĳ�ֵ */
			CorrectChordInterp(tmpPoints, 3, Knots, Param);
		}
		else {
			/*���� RefinedCentripetal �Ĳ�ֵ*/
			RefinedCentripetalInterp(tmpPoints, 3, Knots, Param);
		}

		for (int i = 0; i < Param.size(); i++)
		{
			LocalParam[k].push_back(Param[i]);
		}

		//std::cout << min_i << " ";

	}

	std::vector<double> InteralPram;  //����ÿһ���м��ľֲ�����

	//std::cout << n - 3 << "\n";

	for (int i = 0; i < n - 2; i++)
	{
		//std::cout << i << "\n";

		if (i == 0) {
			InteralPram.push_back(LocalParam[i][1] / LocalParam[i][2]);
		}
		else if (i == n - 3) {
			InteralPram.push_back((LocalParam[i - 1][2] - LocalParam[i - 1][1]) / (LocalParam[i - 1][3] - LocalParam[i - 1][1]));
		}
		else {
			double s1, s2;
			s1 = (LocalParam[i - 1][2] - LocalParam[i - 1][1]) / (LocalParam[i - 1][3] - LocalParam[i - 1][1]);
			s2 = LocalParam[i][1] / LocalParam[i][2];

			InteralPram.push_back((s1 + s2) / 2);
		}

		//std::cout << InteralPram[i] << " ";

	}

	vector<double> x(n - 1);
	x[0] = 1;
	//std::cout << "\n";
	for (int i = 0; i < x.size(); i++)
	{
		if (i > 0)
			x[i] = ((1 - InteralPram[i - 1]) * x[i - 1]) / InteralPram[i - 1];
		//std::cout << x[i] << " ";
	}

	double sum = 0;
	//std::cout << "\n";
	for (int i = 0; i < x.size(); i++)
	{
		sum += x[i];
		x[i] = sum;
		//std::cout << x[i] << " ";
	}

	param[0] = 0, param[n - 1] = 1;

	for (int i = 1; i < n - 1; i++) {
		param[i] = x[i - 1] / sum;
	}

	double num = m - 2 * degree - 2;

	for (int i = 0; i <= degree; i++)knots[i] = 0;
	for (int i = degree + 1; i < degree + num + 1; i++) {
		double sum = 0;
		for (int k = i - degree; k <= i - 1; k++) {
			sum += param[k];
		}
		knots[i] = sum / (double)degree;

		//std::cout << knots[i] << std::endl;
	}
	for (int i = degree + num + 1; i < m; i++)knots[i] = 1;

}



void ClassfierInterp3(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param, std::vector<int> minis) {

	int n = points.size();
	int m = n + degree + 1;
	knots.resize(m);
	param.resize(n);

	std::vector<std::vector<double>> LocalParam; //����ÿһ�εĲ���������
	LocalParam.resize(points.size() - 3);

	std::vector<glm::dvec3> tmpPoints;
	tmpPoints.resize(4);

	std::vector<glm::dvec3> pts; //Ϊ������ͼ��㷽��
	std::vector<std::vector<double>>  Knots; //�ڵ�
	std::vector<std::vector<double>> Param; //����

	for (int k = 0; k < LocalParam.size(); k++)
	{

		/*
		std::vector<std::vector<double>>  Knots; //�ڵ�
		std::vector<std::vector<double>> Param; //����
		Knots.resize(3);
		Param.resize(3);
		*/

		std::vector<double> Knots;
		std::vector<double> Param;

		tmpPoints[0] = points[k];
		tmpPoints[1] = points[k + 1];
		tmpPoints[2] = points[k + 2];
		tmpPoints[3] = points[k + 3];

		int min_i = minis[k] - 4;
		if (min_i == 0) {
			/*���� Universal �Ĳ�ֵ */
			UniversalInterp(tmpPoints, 3, Knots, Param);
		}
		else if (min_i == 1) {
			/*���� CorrectChord �Ĳ�ֵ */
			CorrectChordInterp(tmpPoints, 3, Knots, Param);
		}
		else {
			/*���� RefinedCentripetal �Ĳ�ֵ*/
			RefinedCentripetalInterp(tmpPoints, 3, Knots, Param);
		}

		for (int i = 0; i < Param.size(); i++)
		{
			LocalParam[k].push_back(Param[i]);
		}

		//std::cout << min_i << " ";

	}

	std::vector<double> InteralPram;  //����ÿһ���м��ľֲ�����

	//std::cout << n - 3 << "\n";

	for (int i = 0; i < n - 2; i++)
	{
		//std::cout << i << "\n";

		if (i == 0) {
			InteralPram.push_back(LocalParam[i][1] / LocalParam[i][2]);
		}
		else if (i == n - 3) {
			InteralPram.push_back((LocalParam[i - 1][2] - LocalParam[i - 1][1]) / (LocalParam[i - 1][3] - LocalParam[i - 1][1]));
		}
		else {
			double s1, s2;
			s1 = (LocalParam[i - 1][2] - LocalParam[i - 1][1]) / (LocalParam[i - 1][3] - LocalParam[i - 1][1]);
			s2 = LocalParam[i][1] / LocalParam[i][2];

			InteralPram.push_back((s1 + s2) / 2);
		}

		//std::cout << InteralPram[i] << " ";

	}

	vector<double> x_f(n - 1);
	x_f[0] = 1;
	//std::cout << "\n";
	for (int i = 0; i < x_f.size(); i++)
	{
		if (i > 0)
			x_f[i] = ((1 - InteralPram[i - 1]) * x_f[i - 1]) / InteralPram[i - 1];
		//std::cout << x[i] << " ";
	}

	vector<double> x_b(n - 1);
	x_b[n - 2] = 1;

	for (int i = n-3; i >= 0; i--)
	{
		x_b[i] = InteralPram[i] * x_b[i + 1] / (1 - InteralPram[i]);
	}


	vector<double> x(n - 1);
	//std::cout << "\n";
	for (int i = 0; i < x.size(); i++)
	{
		x[i] = (x_f[i] * x_b[i]) / 2;
		//std::cout << x[i] << " ";
	}

	double sum = 0;
	//std::cout << "\n";
	for (int i = 0; i < x.size(); i++)
	{
		sum += x[i];
		x[i] = sum;
		//std::cout << x[i] << " ";
	}

	param[0] = 0, param[n - 1] = 1;

	for (int i = 1; i < n - 1; i++) {
		param[i] = x[i - 1] / sum;
	}

	double num = m - 2 * degree - 2;

	for (int i = 0; i <= degree; i++)knots[i] = 0;
	for (int i = degree + 1; i < degree + num + 1; i++) {
		double sum = 0;
		for (int k = i - degree; k <= i - 1; k++) {
			sum += param[k];
		}
		knots[i] = sum / (double)degree;

		//std::cout << knots[i] << std::endl;
	}
	for (int i = degree + num + 1; i < m; i++)knots[i] = 1;

}




void ClassfierInterp4(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param, std::vector<int> minis) {

	int n = points.size();
	int m = n + degree + 1;
	knots.resize(m);
	param.resize(n);

	std::vector<std::vector<double>> LocalParam; //����ÿһ�εĲ���������
	LocalParam.resize(points.size() - 3);

	std::vector<glm::dvec3> tmpPoints;
	tmpPoints.resize(4);

	std::vector<glm::dvec3> pts; //Ϊ������ͼ��㷽��
	std::vector<std::vector<double>>  Knots; //�ڵ�
	std::vector<std::vector<double>> Param; //����

	for (int k = 0; k < LocalParam.size(); k++)
	{

		/*
		std::vector<std::vector<double>>  Knots; //�ڵ�
		std::vector<std::vector<double>> Param; //����
		Knots.resize(3);
		Param.resize(3);
		*/

		std::vector<double> Knots;
		std::vector<double> Param;

		tmpPoints[0] = points[k];
		tmpPoints[1] = points[k + 1];
		tmpPoints[2] = points[k + 2];
		tmpPoints[3] = points[k + 3];

		int min_i = minis[k] - 4;
		if (min_i == 0) {
			/*���� Universal �Ĳ�ֵ */
			UniversalInterp(tmpPoints, 3, Knots, Param);
		}
		else if (min_i == 1) {
			/*���� CorrectChord �Ĳ�ֵ */
			CorrectChordInterp(tmpPoints, 3, Knots, Param);
		}
		else {
			/*���� RefinedCentripetal �Ĳ�ֵ*/
			RefinedCentripetalInterp(tmpPoints, 3, Knots, Param);
		}

		for (int i = 0; i < Param.size(); i++)
		{
			LocalParam[k].push_back(Param[i]);
		}

		//std::cout << min_i << " ";

	}

	std::vector<double> InteralPram;  //����ÿһ���м��ľֲ�����

	//std::cout << n - 3 << "\n";

	for (int i = 0; i < n - 2; i++)
	{
		//std::cout << i << "\n";

		if (i == 0) {
			InteralPram.push_back(LocalParam[i][1] / LocalParam[i][2]);
		}
		else if (i == n - 3) {
			InteralPram.push_back((LocalParam[i - 1][2] - LocalParam[i - 1][1]) / (LocalParam[i - 1][3] - LocalParam[i - 1][1]));
		}
		else {
			double s1, s2;
			s1 = (LocalParam[i - 1][2] - LocalParam[i - 1][1]) / (LocalParam[i - 1][3] - LocalParam[i - 1][1]);
			s2 = LocalParam[i][1] / LocalParam[i][2];

			InteralPram.push_back((s1 + s2) / 2);
		}

		//std::cout << InteralPram[i] << " ";

	}

	vector<double> x_f(n - 1);
	x_f[0] = 1;
	//std::cout << "\n";
	for (int i = 0; i < x_f.size(); i++)
	{
		if (i > 0)
			x_f[i] = ((1 - InteralPram[i - 1]) * x_f[i - 1]) / InteralPram[i - 1];
		//std::cout << x[i] << " ";
	}

	vector<double> x_b(n - 1);
	x_b[n - 2] = 1;

	for (int i = n - 3; i >= 0; i--)
	{
		x_b[i] = InteralPram[i] * x_b[i + 1] / (1 - InteralPram[i]);
	}


	vector<double> x(n - 1);
	//std::cout << "\n";
	for (int i = 0; i < x.size(); i++)
	{
		x[i] = (x_f[i] * x_b[i]) / 2;
		//std::cout << x[i] << " ";
	}

	double sum = 0;
	//std::cout << "\n";
	for (int i = 0; i < x.size(); i++)
	{
		sum += x[i];
		x[i] = sum;
		//std::cout << x[i] << " ";
	}

	param[0] = 0, param[n - 1] = 1;

	for (int i = 1; i < n - 1; i++) {
		param[i] = x[i - 1] / sum;
	}

	double num = m - 2 * degree - 2;

	for (int i = 0; i <= degree; i++)knots[i] = 0;
	for (int i = degree + 1; i < degree + num + 1; i++) {
		double sum = 0;
		for (int k = i - degree; k <= i - 1; k++) {
			sum += param[k];
		}
		knots[i] = sum / (double)degree;

		//std::cout << knots[i] << std::endl;
	}
	for (int i = degree + num + 1; i < m; i++)knots[i] = 1;

}




void ModifiedChordInterp(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param) {

	int n = points.size();
	int m = n + degree + 1;
	param.resize(n);


	vector<double> ChordLen;
	vector<double> MChordLen;

	for (int i = 1; i < n; i++)
	{
		double len = sqrt(pow((points[i].x - points[i - 1].x), 2) + pow((points[i].y - points[i - 1].y), 2));

		ChordLen.push_back(len);
	}

	glm::dvec2 last, current;  //���浥λ����Ľ�ƽ���߷���

	for (int i = 1; i < n; i++)
	{
		if (i==1)
		{
			glm::dvec2 a, b, c;

			a.x = points[i].x - points[i - 1].x;
			a.y = points[i].y - points[i - 1].y;

			b.x = points[i + 1].x - points[i].x;
			b.y = points[i + 1].y - points[i].y;

			double tmp;
			tmp = sqrt(pow(a.x, 2) + pow(a.y, 2));
			a.x /= (tmp + std::numeric_limits<double>::epsilon());
			a.y /= (tmp + std::numeric_limits<double>::epsilon());

			tmp = sqrt(pow(b.x, 2) + pow(b.y, 2));
			b.x /= (tmp + std::numeric_limits<double>::epsilon());
			b.y /= (tmp + std::numeric_limits<double>::epsilon());

			if (fabs(a.x * b.y - b.x * a.y) < 1e-4) {  //���ֹ��ߵ����
				last = b;
				current = b;
				MChordLen.push_back(ChordLen[i-1]);
				continue;
			}

			c.x = a.x + b.x;
			c.y = a.y + b.y;
			tmp = sqrt(pow(c.x, 2) + pow(c.y, 2));
			c.x /= (tmp + std::numeric_limits<double>::epsilon());
			c.y /= (tmp + std::numeric_limits<double>::epsilon());

			last = c;
			current = c;

			double angle;
			if (fabs(a.x - c.x) < 1e-4 && fabs(a.y - c.y) < 1e-4) {
				angle = 0;
			}
			else
			{
				angle = std::acos(a.x * c.x + a.y * c.y);
				if (angle > M_PI_2)
				{
					angle = M_PI - angle;
				}
			}

			double centralangle;
			if (M_PI - angle * 2 > M_PI_2) {
				centralangle = angle * 2;
			}
			else
			{
				centralangle = M_PI - angle * 2;
			}
			MChordLen.push_back(ChordLen[i - 1] / 2.0 / (sin(centralangle / 2.0) + std::numeric_limits<double>::epsilon()) * centralangle);
		}
		else if(i == n-1)
		{
			glm::dvec2 a, c;
			a.x = points[i].x - points[i - 1].x;
			a.y = points[i].y - points[i - 1].y;

			double tmp;
			tmp = sqrt(pow(a.x, 2) + pow(a.y, 2));
			a.x /= (tmp + std::numeric_limits<double>::epsilon());
			a.y /= (tmp + std::numeric_limits<double>::epsilon());

			c = current;

			//std::cout << a.x << " " << a.y << "\n";

			//std::cout << c.x << " " << c.y << "\n";

			double angle;
			if (fabs(a.x - c.x) < 1e-4 && fabs(a.y - c.y) < 1e-4) {
				angle = 0;
				MChordLen.push_back(ChordLen[i - 1]);
				continue;
			}
			else
			{
				angle = std::acos(a.x * c.x + a.y * c.y);
				if (angle > M_PI_2)
				{
					angle = M_PI - angle;
				}
			}

			//std::cout << angle << "\n";

			double centralangle;
			if (M_PI - angle * 2 > M_PI_2) {
				centralangle = angle * 2;
			}
			else
			{
				centralangle = M_PI - angle * 2;
			}

			//std::cout << centralangle << "\n";

			MChordLen.push_back(ChordLen[i - 1] / 2.0 / (sin(centralangle / 2.0) + std::numeric_limits<double>::epsilon()) * centralangle);

		}
		else
		{
			last = current;

			glm::dvec2 a, b, c;

			a.x = points[i].x - points[i - 1].x;
			a.y = points[i].y - points[i - 1].y;

			b.x = points[i + 1].x - points[i].x;
			b.y = points[i + 1].y - points[i].y;

			double tmp;
			tmp = sqrt(pow(a.x, 2) + pow(a.y, 2));
			a.x /= (tmp + std::numeric_limits<double>::epsilon());
			a.y /= (tmp + std::numeric_limits<double>::epsilon());

			tmp = sqrt(pow(b.x, 2) + pow(b.y, 2));
			b.x /= (tmp + std::numeric_limits<double>::epsilon());
			b.y /= (tmp + std::numeric_limits<double>::epsilon());
			
			c.x = a.x + b.x;
			c.y = a.y + b.y;
			tmp = sqrt(pow(c.x, 2) + pow(c.y, 2));
			c.x /= (tmp + std::numeric_limits<double>::epsilon());
			c.y /= (tmp + std::numeric_limits<double>::epsilon());

			current = c;

			double angle;

			if (fabs(a.x - c.x) < 1e-4 && fabs(a.y - c.y) < 1e-4) { //����
				angle = 0;
				MChordLen.push_back(ChordLen[i - 1]);
				continue;

			}else if (fabs(a.x - last.x) < 1e-4 && fabs(a.y - last.y) < 1e-4) { //����
				angle = 0;
				MChordLen.push_back(ChordLen[i - 1]);
				continue;

			}else if (fabs(last.x - current.x) < 1e-4 && fabs(last.y - current.y) < 1e-4) {//ƽ���ߵ����
				angle = 0;
				MChordLen.push_back(ChordLen[i - 1]);
				continue;
			}
			else
			{
				angle = std::acos(last.x * current.x + last.y * current.y);
				if (angle > M_PI_2)
				{
					angle = M_PI - angle;
				}
			}

			double centralangle=angle;
			/*if (M_PI - angle * 2 > M_PI_2) {
				centralangle = angle * 2;
			}
			else
			{
				centralangle = M_PI - angle * 2;
			}*/
			MChordLen.push_back(ChordLen[i - 1] / 2.0 / (sin(centralangle / 2.0) + std::numeric_limits<double>::epsilon()) * centralangle);
		}


	}

	double sum = 0;
	vector<double> accMChordLen;

	for (int i = 0; i < n-1; i++) {
		sum += MChordLen[i];
		accMChordLen.push_back(sum);
	}

	param[0] = 0, param[n - 1] = 1;
	for (int i = 1; i < n - 1; i++) {
		param[i] = accMChordLen[i - 1] / (sum + std::numeric_limits<double>::epsilon());
	}

	/*double num = m - 2 * degree - 2;
	knots.resize(m);
	for (int i = 0; i <= degree; i++)knots[i] = 0;
	for (int i = degree + 1; i < degree + num + 1; i++) {
		double sum = 0;
		for (int k = i - degree; k <= i - 1; k++) {
			sum += param[k];
		}
		knots[i] = sum / (double)degree;
	}
	for (int i = degree + num + 1; i < m; i++)knots[i] = 1;*/
#ifdef KTP
	KTPKnots(degree, knots, param);
#elif defined(UNIFORM_KNOTS)
	UniformKnots(degree, knots, param);
#elif defined(NATURAL)
	NaturalKnots(degree, knots, param);
#else
	AvgKnots(degree, knots, param);
#endif


}




void ZCMInterp(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param) {

	int n = points.size();
	int m = n + degree + 1;
	param.resize(n);

	vector<double> v, w;

	for (int i = 0; i <= n - 4; i++)
	{
		double d = (points[i + 3].x - points[i + 2].x) * (points[i + 1].y - points[i + 2].y) -
			(points[i + 1].x - points[i + 2].x) * (points[i + 3].y - points[i + 2].y);

		double v_tmp = (points[i + 1].y - points[i + 2].y) * (points[i].x - points[i + 2].x) / d +
			(points[i + 2].x - points[i + 1].x) * (points[i].y - points[i + 2].y) / d;

		double w_tmp = (points[i + 2].y - points[i + 3].y) * (points[i].x - points[i + 2].x) / d +
			(points[i + 3].x - points[i + 2].x) * (points[i].y - points[i + 2].y) / d;

		v.push_back(v_tmp);
		w.push_back(w_tmp);
	}

	vector<double> s_r; //对应s_mao，需要从下标2开始，如果共线或者不满足凸包，则取值设为-1
	s_r.push_back(-1);
	s_r.push_back(-1);

	for (int i = 2; i <= n - 2; i++)
	{
		//先判断是否共线
		glm::dvec2 n1, n2;
		n1.x = points[i].x - points[i - 1].x;
		n1.y = points[i].y - points[i - 1].y;
		n2.x = points[i + 1].x - points[i].x;
		n2.y = points[i + 1].y - points[i].y;
		if (fabs(n1.x * n2.y - n1.y * n2.x) < 1e-4) {
			s_r.push_back(-1);
			continue;
		}

		if (v[i - 2] > 1e-4 && w[i - 2] > 1.0 && fabs(w[i - 2] - 1.0) > 1e-4) {
			double tmp_s_r = (v[i - 2] + sqrt(v[i - 2] * w[i - 2] / (v[i - 2] + w[i - 2] - 1))) / (v[i - 2] + w[i - 2]);
			s_r.push_back(tmp_s_r);
		}
		else
		{
			s_r.push_back(-1);
		}
	}

	vector<double> Chord_length; //计算每一段弦长
	for (int i = 1; i < n; i++)
	{
		glm::dvec2 n1;
		n1.x = points[i].x - points[i - 1].x;
		n1.y = points[i].y - points[i - 1].y;
		Chord_length.push_back(sqrt(pow(n1.x, 2) + pow(n1.y, 2)));
	}


	vector<double> Local_Param;

	for (int i = 1; i < n - 1; i++)
	{
		if (i == 1) {
			if (s_r[i + 1] > 1e-4) {
				double tao = v[i - 1] - (v[i - 1] + w[i - 1]) * s_r[i + 1] + s_r[i + 1];
				double tmp_s_l = tao / (tao - s_r[i + 1]);
				Local_Param.push_back(tmp_s_l);
			}
			else
			{
				double tmp_chord = sqrt(Chord_length[i - 1]) / (sqrt(Chord_length[i - 1]) + sqrt(Chord_length[i]));
				Local_Param.push_back(tmp_chord);
			}
		}
		else if (i == n - 2) {
			if (s_r[i] > 1e-4) {
				Local_Param.push_back(s_r[i]);
			}
			else {
				double tmp_chord = sqrt(Chord_length[i - 1]) / (sqrt(Chord_length[i - 1]) + sqrt(Chord_length[i]));
				Local_Param.push_back(tmp_chord);
			}
		}
		else {
			if (s_r[i] > 1e-4 && s_r[i + 1] > 1e-4) {
				double tao = v[i - 1] - (v[i - 1] + w[i - 1]) * s_r[i + 1] + s_r[i + 1];
				double tmp_s_l = tao / (tao - s_r[i + 1]);
				Local_Param.push_back((s_r[i] + tmp_s_l) / 2);
			}
			else if (s_r[i] > 1e-4) {
				Local_Param.push_back(s_r[i]);
			}
			else if (s_r[i + 1] > 1e-4) {
				double tao = v[i - 1] - (v[i - 1] + w[i - 1]) * s_r[i + 1] + s_r[i + 1];
				double tmp_s_l = tao / (tao - s_r[i + 1]);
				Local_Param.push_back(tmp_s_l);
			}
			else
			{
				double tmp_chord = sqrt(Chord_length[i - 1]) / (sqrt(Chord_length[i - 1]) + sqrt(Chord_length[i]));
				Local_Param.push_back(tmp_chord);
			}
		}
	}


	vector<double> InteralPram;

	//std::cout << "ZCM: " << std::endl;
	for (int i = 0; i < Local_Param.size(); i++)
	{
		//std::cout << Local_Param[i] << " ";
		InteralPram.push_back(Local_Param[i]);
	}
	//std::cout << std::endl;


	double delta2, deltan;
	delta2 = 1;
	deltan = 1;

	vector<double> delta;
	delta.resize(n - 1);
	double sum = 0;

	if (n > 4) {

		std::vector<double> f, h;

		//h: 2 - n-1
		//f: 3 - n-1
		for (int i = 0; i < InteralPram.size(); i++)
		{
			h.push_back(InteralPram[i] * (1 - InteralPram[i]));

			if (i > 0)
				f.push_back(InteralPram[i - 1] * InteralPram[i - 1] + (1 - InteralPram[i]) * (1 - InteralPram[i]));
		}


		Eigen::MatrixXd mat(n - 3, n - 3);

		Eigen::VectorXd res(n - 3);

		Eigen::VectorXd x(n - 3);


		for (int i = 0; i < mat.rows(); i++)
		{
			if (i == 0) {
				mat(i, 0) = f[0];
				mat(i, 1) = -h[1];
				for (int j = 2; j < mat.cols(); j++) {
					mat(i, j) = 0;
				}
			}
			else if (i == n - 4)
			{
				for (int j = 0; j < mat.cols() - 2; j++) {
					mat(i, j) = 0;
				}
				mat(i, mat.cols() - 2) = -h[h.size() - 2];
				mat(i, mat.cols() - 1) = f[f.size() - 1];
			}
			else
			{
				for (int j = 0; j < i - 1; j++) {
					mat(i, j) = 0;
				}

				mat(i, i - 1) = -h[i];
				mat(i, i) = f[i];
				mat(i, i + 1) = -h[i + 1];

				for (int j = i + 2; j < mat.cols(); j++) {
					mat(i, j) = 0;
				}
			}
		}

		for (int i = 0; i < res.size(); i++)
		{
			if (i == 0) {
				res(i) = h[0] * delta2;
			}
			else if (i == res.size() - 1)
			{
				res(i) = h[h.size() - 1] * deltan;

			}
			else
			{
				res(i) = 0;
			}

		}		

		/*std::cout << "ZCM: " << std::endl;

		for (int i = 0; i < Chord_length.size(); i++)
		{
			std::cout << Chord_length[i] << " ";
		}
		std::cout << std::endl;

		for (int i = 0; i < Local_Param.size(); i++)
		{
			std::cout << Local_Param[i] << " ";
		}
		std::cout << std::endl;

		for (int i = 0; i < n - 3; i++)
		{
			for (int j = 0; j < n - 3; j++)
			{
				std::cout << mat(i, j) << " ";
			}
			std::cout << endl;
		}


		for (int i = 0; i < n - 3; i++)
		{
			std::cout << res(i) << " ";
		}
		std::cout << endl;*/



		//普通的QR分解求解
		x = mat.colPivHouseholderQr().solve(res);

		//x = mat.llt().solve(res);

		/*
		//三对角矩阵求解
		Eigen::Tridiagonalization<Eigen::MatrixXd> tri(mat);
		x = tri.solve(res);
		*/
		for (int i = 0; i < delta.size(); i++)
		{
			if (i == 0)
				delta[i] = delta2;
			else if (i == delta.size() - 1)
			{
				delta[i] = deltan;
			}
			else {
				delta[i] = x[i - 1];
			}
			sum += delta[i];

			delta[i] = sum;

		}
	}
	else {
		for (int i = 0; i < delta.size(); i++)
		{
			if (i == 0)
				delta[i] = delta2;
			else if (i == delta.size() - 1)
			{
				delta[i] = deltan;
			}
			else {
				delta[i] = ((2 * InteralPram[0] * (1 - InteralPram[0])) + (2 * InteralPram[1] * (1 - InteralPram[1]))) /
					(2 * InteralPram[0] * InteralPram[0] + 2 * (1 - InteralPram[1]) * (1 - InteralPram[1]));
			}
			sum += delta[i];
			delta[i] = sum;
		}
	}

	param[0] = 0, param[n - 1] = 1;

	for (int i = 1; i < n - 1; i++) {
		param[i] = delta[i - 1] / sum;
	}

	/*double num = m - 2 * degree - 2;
	knots.resize(m);
	for (int i = 0; i <= degree; i++)knots[i] = 0;
	for (int i = degree + 1; i < degree + num + 1; i++) {
		double sum = 0;
		for (int k = i - degree; k <= i - 1; k++) {
			sum += param[k];
		}
		knots[i] = sum / (double)degree;
	}
	for (int i = degree + num + 1; i < m; i++)knots[i] = 1;*/

#ifdef KTP
	KTPKnots(degree, knots, param);
#elif defined(UNIFORM_KNOTS)
	UniformKnots(degree, knots, param);
#elif defined(NATURAL)
	NaturalKnots(degree, knots, param);
#else
	AvgKnots(degree, knots, param);
#endif

}