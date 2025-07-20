#pragma once
#include <Eigen/Core>
#include <vector>
#include <map>

using namespace std;
using namespace Eigen;

typedef std::pair<int, double> Parameter;

typedef enum { RECT_INIT, SPHERE_INIT } EInitType;

// compute the x ,y position of current parameter
Vector2d getPos(const Parameter& para,const vector<Vector2d>& controls_) ;

// compute the first differential
Vector2d getFirstDiff(const Parameter& para, const vector<Vector2d>& controls_) ;

// compute the second differential
Vector2d getSecondDiff(const Parameter& para, const vector<Vector2d>& controls_) ;

// compute the curvature
double getCurvature(const Parameter& para, const vector<Vector2d>& controls_) ;

// compute the unit tangent vector
Vector2d getTangent(const Parameter& para, const vector<Vector2d>& controls_);

// compute the unit Normal vector
Vector2d getNormal(const Parameter& para, const vector<Vector2d>& controls_);

// compute the Curvature center ( rho = k)
Vector2d getCurvCenter(const Parameter& para, const vector<Vector2d>& controls_);

// compute the foot print
double findFootPrint(const vector<Vector2d>& givepoints,vector<Parameter>& footPrints, const vector<Vector2d>& controls_, const vector<Vector2d>& positions_, double interal_);

// find the coff vector
VectorXd getCoffe(const Parameter& para, const vector<Vector2d>& controls_);

// set the control points and compute a uniform spatial partition of the data points
void setNewControl(vector<Vector2d>& controls_, const vector<Vector2d>& controlPs, double interal_, vector<Vector2d>& positions_);

// check if two point is on same side. para is foot print of p1
bool checkSameSide(Vector2d p1, Vector2d p2, Vector2d neip);

MatrixXd getSIntegralSq(const vector<Vector2d>& controls_);

MatrixXd getFIntegralSq(const vector<Vector2d>& controls_);

Parameter getPara(int index, const vector<Vector2d>& controls_, const vector<Vector2d>& positions_, double interal_);

double  geo_approximation(const vector<Vector2d>& points,
	//CubicBSplineCurve& curve,
	vector<Vector2d>& controls_,
	vector<Vector2d>& positions_,
	double interal_ = 0.001,
	int controlNum = 11,
	int maxIterNum = 30,
	double alpha = 0.002,
	double beta = 0.005,  // 0.005	     
	double eplison = 0.0001,
	EInitType initType = SPHERE_INIT);

void initControlPoint(const vector<Vector2d>& points,
	vector<Vector2d>& controlPs,
	int controlNum,
	EInitType initType);





