#include"geo_Approximation.h"
#include <eigen/SVD>
#include <fstream>
#include <iostream>

#include <ANN/ANN.h>

#define TEMP_PI       3.14159265358979323846

Vector2d getPos(const Parameter& para, const vector<Vector2d>& controls_)
{
	MatrixXd cm(4, 4);
	cm << -1, 3, -3, 1,
		3, -6, 3, 0,
		-3, 0, 3, 0,
		1, 4, 1, 0;

	double tf = para.second;
	int ki = para.first;

	MatrixXd  tm(1, 4);
	tm << tf * tf * tf, tf* tf, tf, 1;


	size_t n = controls_.size();
	MatrixXd pm(4, 2);
	for (int i = 0; i < 4; i++)
	{
		pm(i, 0) = controls_[(ki + i) % n].x() / 6.0;
		pm(i, 1) = controls_[(ki + i) % n].y() / 6.0;
	}
	MatrixXd rm = tm * cm * pm;

	return Vector2d(rm(0, 0), rm(0, 1));
}

void setNewControl(vector<Vector2d>& controls_, const vector<Vector2d>& controlPs, double interal_, vector<Vector2d>& positions_)
{
	controls_.clear();
	positions_.clear();
	controls_ = controlPs;

	for (unsigned int i = 0; i < controls_.size(); i++)
	{
		for (double fj = 0; fj <= 1.0f; fj += interal_)
		{
			Parameter temp(i, fj);
			Vector2d p = getPos(temp, controls_);
			positions_.push_back(p);
		}
	}
}


Vector2d getFirstDiff(const Parameter& para, const vector<Vector2d>& controls_)
{
	MatrixXd cm(4, 4);
	cm << -1, 3, -3, 1,
		3, -6, 3, 0,
		-3, 0, 3, 0,
		1, 4, 1, 0;

	double tf = para.second;
	int ki = para.first;

	MatrixXd  tm(1, 4);
	tm << 3 * tf * tf, 2 * tf, 1, 0;

	size_t n = controls_.size();
	MatrixXd pm(4, 2);
	for (int i = 0; i < 4; i++)
	{
		pm(i, 0) = controls_[(ki + i) % n].x() / 6.0f;
		pm(i, 1) = controls_[(ki + i) % n].y() / 6.0f;
	}


	MatrixXd rm = tm * cm * pm;

	return Vector2d(rm(0, 0), rm(0, 1));

}


Vector2d getSecondDiff(const Parameter& para, const vector<Vector2d>& controls_)
{
	MatrixXd cm(4, 4);
	cm << -1, 3, -3, 1,
		3, -6, 3, 0,
		-3, 0, 3, 0,
		1, 4, 1, 0;


	double tf = para.second;
	int ki = para.first;
	MatrixXd  tm(1, 4);
	tm << 6 * tf, 2, 0, 0;

	int n = controls_.size();
	MatrixXd pm(4, 2);
	for (int i = 0; i < 4; i++)
	{
		pm(i, 0) = controls_[(ki + i) % n].x() / 6.0;
		pm(i, 1) = controls_[(ki + i) % n].y() / 6.0;
	}
	MatrixXd rm = tm * cm * pm;

	return Vector2d(rm(0, 0), rm(0, 1));

}


double getCurvature(const Parameter& para, const vector<Vector2d>& controls_)
{
	Vector2d fp = getFirstDiff(para, controls_);
	Vector2d sp = getSecondDiff(para, controls_);

	double kappa = abs(fp.x() * sp.y() - sp.x() * fp.y());
	kappa = kappa / sqrt(pow((fp.x() * fp.x() + fp.y() * fp.y()), 3));

	return kappa;
}


Vector2d getTangent(const Parameter& para, const vector<Vector2d>& controls_)
{
	Vector2d p = getFirstDiff(para, controls_);
	return p.normalized();

}

Vector2d getNormal(const Parameter& para, const vector<Vector2d>& controls_)
{
	Vector2d v = getTangent(para, controls_);
	return Vector2d(-v.y(), v.x());

}


Vector2d getCurvCenter(const Parameter& para, const vector<Vector2d>& controls_)
{
	Vector2d p = getPos(para, controls_);

	Vector2d fd = getFirstDiff(para, controls_);
	Vector2d sd = getSecondDiff(para, controls_);

	double p1 = (fd.x() * fd.x() + fd.y() * fd.y()) * fd.y();
	double p2 = sd.y() * fd.x() - sd.x() * fd.y();
	double alpha = p.x() - p1 / p2;

	double p3 = (fd.x() * fd.x() + fd.y() * fd.y()) * fd.x();
	double beta = p.y() + p3 / p2;


	return Vector2d(alpha, beta);

}

Parameter getPara(int index, const vector<Vector2d>& controls_, const vector<Vector2d>& positions_, double interal_)
{
	int num = (int)(positions_.size() / controls_.size());
	int ki = index / num;
	double tf = interal_ * (index - ki * num);
	return make_pair(ki, tf);
}


double findFootPrint(const vector<Vector2d>& givepoints, vector<Parameter>& footPrints, const vector<Vector2d>& controls_, const vector<Vector2d>& positions_, double interal_)
{
	footPrints.clear();
	footPrints.resize(givepoints.size(), Parameter(0, 0.0));

	int iKNei = 1;
	int iDim = 2;
	size_t iNPts = positions_.size();
	double eps = 0;

	ANNpointArray dataPts = annAllocPts(iNPts, iDim); // allocate data points; // data points
	ANNpoint queryPt = annAllocPt(iDim);  // allocate query point

	ANNidxArray nnIdx = new ANNidx[iKNei]; // allocate near neigh indices
	ANNdistArray dists = new ANNdist[iKNei]; // allocate near neighbor dists

	for (int i = 0; i != iNPts; ++i) {
		dataPts[i][0] = positions_[i].x();
		dataPts[i][1] = positions_[i].y();
	}
	ANNkd_tree* kdTree = new ANNkd_tree( // build search structure
		dataPts, // the data points
		iNPts, // number of points
		iDim);

	double squareSum = 0.0;
	for (int i = 0; i != (int)givepoints.size(); ++i) {
		queryPt[0] = givepoints[i].x();
		queryPt[1] = givepoints[i].y();
		kdTree->annkSearch( // search
			queryPt, // query point
			iKNei, // number of near neighbors
			nnIdx, // nearest neighbors (returned)
			dists, // distance (returned)
			eps); // error bound
		squareSum += dists[0];
		footPrints[i] = getPara(nnIdx[0], controls_, positions_, interal_);
	}

	delete[] nnIdx;
	delete[] dists;
	delete kdTree;
	annClose(); // done with ANN

	return squareSum;
}


MatrixXd getSIntegralSq(const vector<Vector2d>& controls_)
{
	// compute P"(t)
	int controlNum = controls_.size();
	MatrixXd pm(2 * controlNum, 2 * controlNum);
	pm.setZero();

	Matrix2d tIntergrated;
	tIntergrated << 1 / 3.0, 1 / 2.0, 1 / 2.0, 1.0;
	Matrix2d tm;
	tm << 6, 0, 0, 2;


	MatrixXd cm(2, 4);
	cm << -1, 3, -3, 1,
		3, -6, 3, 0;
	cm = cm / 6.0;


	Matrix4d coffm = cm.transpose() * tm.transpose() * tIntergrated * tm * cm;
	for (int i = 0; i < controlNum; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			for (int n = 0; n < 4; n++)
			{
				int kj = (i + j) % controlNum;
				int kn = (i + n) % controlNum;
				pm(kj, kn) += coffm(j, n);
				pm(controlNum + kj, controlNum + kn) += coffm(j, n);
			}
		}
	}
	return pm;
}


MatrixXd getFIntegralSq(const vector<Vector2d>& controls_)
{
	// compute P"(t)
	size_t controlNum = controls_.size();
	MatrixXd pm(2 * controlNum, 2 * controlNum);
	pm.setZero();


	Matrix3d tIntergrated;
	tIntergrated << 1 / 5.0, 1 / 4.0, 1 / 3.0,
		1 / 4.0, 1 / 3.0, 1 / 2.0,
		1 / 3.0, 1 / 2.0, 1 / 1.0;

	MatrixXd cm(3, 4);
	cm << -1, 3, -3, 1,
		3, -6, 3, 0,
		-3, 0, 3, 0;
	cm = cm / 6.0;

	Matrix3d tm;
	tm << 3, 0, 0, 0, 2, 0, 0, 0, 1;

	Matrix4d coffm = cm.transpose() * tm.transpose() * tIntergrated * tm * cm;
	for (int i = 0; i < controlNum; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			for (int n = 0; n < 4; n++)
			{
				int kj = (i + j) % controlNum;
				int kn = (i + n) % controlNum;
				pm(kj, kn) += coffm(j, n);
				pm(controlNum + kj, controlNum + kn) += coffm(j, n);
			}
		}
	}
	return pm;

}

bool checkSameSide(Vector2d p1, Vector2d p2, Vector2d neip)
{
	Vector2d v1 = p2 - neip;
	Vector2d v2 = p1 - neip;
	bool b = true;

	if (v1.x() * v2.x() + v1.y() * v2.y() < 0)
	{
		b = false;
	}

	return  b;
}


VectorXd getCoffe(const Parameter& para, const vector<Vector2d>& controls_)
{
	int ki = para.first;
	double tf = para.second;

	Matrix4d cm(4, 4);
	cm << -1, 3, -3, 1,
		3, -6, 3, 0,
		-3, 0, 3, 0,
		1, 4, 1, 0;

	MatrixXd  tv(1, 4);
	tv << tf * tf * tf, tf* tf, tf, 1;

	MatrixXd rv = tv * cm;

	VectorXd newv(controls_.size());
	newv.setZero();
	for (int i = 0; i < 4; i++)
	{
		newv[(ki + i) % controls_.size()] = rv(0, i) / 6.0f;
	}
	return newv;
}


void initControlPoint(const vector<Vector2d>& points,
	vector<Vector2d>& controlPs,
	int controlNum,
	EInitType initType)
{
	// compute the initial 12 control points
	controlPs.clear();
	//int perNum = controlNum/4;
	int perNum = 3;

	if (initType == RECT_INIT)
	{
		Vector2d v1 = points[0];
		Vector2d v2 = points[0];

		for (unsigned int i = 0; i != points.size(); ++i) {
			Vector2d v = points[i];
			if (v1.x() > v.x())  v1.x() = v.x();
			if (v1.y() > v.y())  v1.y() = v.y();
			if (v2.x() < v.x())  v2.x() = v.x();
			if (v2.y() < v.y())  v2.y() = v.y();
		}

		Vector2d dir = (v2 - v1) * 0.5;
		Vector2d cen = v1 + dir;

		v1 = cen - 1.05 * dir;
		v2 = cen + 1.05 * dir;

		vector<Vector2d> rets;
		rets.push_back(v1);
		rets.push_back(Vector2d(v1.x(), v2.y()));
		rets.push_back(v2);
		rets.push_back(Vector2d(v2.x(), v1.y()));
		rets.push_back(v1);


		for (int i = 0; i < 4; i++)
		{
			Vector2d p1 = rets[i];
			Vector2d p2 = rets[i + 1];
			for (int j = 0; j < perNum; j++) {
				controlPs.push_back(p1 + (p2 - p1) * j / (double)(perNum));
			}
		}
	}
	else
	{
		Vector2d cen(0, 0);
		for (int i = 0; i < points.size(); i++)
		{
			cen += points[i];
		}
		cen /= points.size();

		double radius = 0;
		for (int i = 0; i < points.size(); i++)
		{
			double len = (points[i] - cen).norm();
			if (radius < len)
				radius = len;
		}

		double theta = (2 * TEMP_PI) / controlNum;
		for (int i = 0; i < controlNum; i++)
		{
			Vector2d pos = cen + radius * Vector2d(std::cos(theta * i), std::sin(theta * i));
			controlPs.push_back(pos);
		}
	}

}

double geo_approximation(
	const vector<Vector2d>& points,
	vector<Vector2d>& controls_,
	vector<Vector2d>& positions_,
	double interal_ ,
	int controlNum /* = 28 */,
	int maxIterNum  /*= 30 */,
	double alpha /* = 0.002 */,
	double gama /* = 0.002 */,
	double eplison /* = 0.0001 */,
	EInitType initType /* =SPHERE_INIT */)
{

	vector<Vector2d> controlPs;
	initControlPoint(points, controlPs, controlNum, initType);

	setNewControl(controls_, controlPs,interal_, positions_);

	// update the control point
	// compute P"(t)
	MatrixXd pm = getSIntegralSq(controls_);
	MatrixXd sm = getFIntegralSq(controls_);
	// end test

	// find the foot print, will result in error
	std::vector< std::pair<int, double> > parameters;
	double fsd = findFootPrint(points, parameters, controls_, positions_, interal_);

	int iterNum = 0;
	while (fsd > eplison && iterNum < maxIterNum)
	{
		MatrixXd ehm(2 * controlNum, 2 * controlNum);
		VectorXd ehv(2 * controlNum);

		ehm.setZero();
		ehv.setZero();

		// compute h(D)
		for (int i = 0; i < (int)parameters.size(); i++)
		{
			// 			if( labels[i] == false )
			// 				continue;

			// compute d, rho, Tkv, Nkv
			double kappa = getCurvature(parameters[i], controls_);
			double rho = 10e+6;
			Vector2d neip = getPos(parameters[i], controls_);
			Vector2d Tkv = getTangent(parameters[i], controls_);
			Vector2d Nkv = getNormal(parameters[i], controls_);

			double d = (points[i] - neip).norm();
			Vector2d Kv(0.0, 0.0);
			bool sign = true;
			if (kappa != 0.0f)
			{
				rho = 1 / kappa;
				Kv = getCurvCenter(parameters[i], controls_);
				double ddd = (Kv - neip).norm();
				sign = checkSameSide(Kv, points[i], neip);
			}

			VectorXd coffv = getCoffe(parameters[i], controls_);


			MatrixXd tempcoffm1(controlNum, 1);
			for (int ij = 0; ij < controlNum; ij++)
				tempcoffm1(ij, 0) = coffv[ij];
			MatrixXd tempcoffm = tempcoffm1 * (tempcoffm1.transpose());

			// update the matrix
			double fxx = Tkv.x() * Tkv.x();
			double fyy = Tkv.y() * Tkv.y();
			double fxy = Tkv.x() * Tkv.y();

			Vector2d oldp = neip - points[i];

			if (!sign)
			{
				d = -d;
				VectorXd tempv1 = (coffv) * (d / (d - rho)) * (fxx * points[i].x() + fxy * points[i].y());
				VectorXd tempv2 = (coffv) * (d / (d - rho)) * (fyy * points[i].y() + fxy * points[i].x());
				for (int i2 = 0; i2 < controlNum; i2++)
				{
					for (int j = 0; j < controlNum; j++)
					{
						double fp = (d / (d - rho)) * tempcoffm(i2, j);
						ehm(i2, j) += fxx * fp;
						ehm(i2, j + controlNum) += fxy * fp;
						ehm(i2 + controlNum, j) += fxy * fp;
						ehm(i2 + controlNum, j + controlNum) += fyy * fp;
					}
					ehv[i2] += tempv1[i2];
					ehv[i2 + controlNum] += tempv2[i2];
				}
			}
			fxx = Nkv.x() * Nkv.x();
			fyy = Nkv.y() * Nkv.y();
			fxy = Nkv.x() * Nkv.y();
			VectorXd tempv1 = (coffv) * (fxx * points[i].x() + fxy * points[i].y());
			VectorXd tempv2 = (coffv) * (fyy * points[i].y() + fxy * points[i].x());
			for (int i2 = 0; i2 < controlNum; i2++)
			{
				for (int j = 0; j < controlNum; j++)
				{
					double fp = tempcoffm(i2, j);
					ehm(i2, j) += fxx * fp;
					ehm(i2, j + controlNum) += fxy * fp;
					ehm(i2 + controlNum, j) += fxy * fp;
					ehm(i2 + controlNum, j + controlNum) += fyy * fp;
				}
				ehv[i2] += tempv1[i2];
				ehv[i2 + controlNum] += tempv2[i2];
			}
		}

		// check if ehm, ehv right
		//solve the function
		MatrixXd fm = ehm * 0.5 + pm * alpha + sm * gama;
		VectorXd ehv2 = ehv * 0.5;
		JacobiSVD<MatrixXd> svd(fm, ComputeThinU | ComputeThinV);
		VectorXd resultxy = svd.solve(ehv2);

		// update the curve
		for (int i = 0; i < controlNum; i++) {
			controlPs[i] = Vector2d(resultxy[i], resultxy[i + controlNum]);
		}

		setNewControl(controls_, controlPs, interal_, positions_);

		++iterNum;

		fsd = findFootPrint(points, parameters, controls_, positions_, interal_);

	}
	return fsd;
}
