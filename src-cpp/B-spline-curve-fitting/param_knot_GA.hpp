#ifndef _PARAM_KNOT_GA_
#define _PARAM_KNOT_GA_
#define REAL_INF std::numeric_limits<double>::infinity()

#include <string>
#include <iomanip>
#include "openGA/src/openGA.hpp"
#include <algorithm>
#include <limits>
#include "alg_Interpolation.h"
#include "CalculateCurve.h"

struct MyPairSolution
{
    tinynurbs::Curve<double> crv;
	std::vector<double> param;
    std::vector<double> knot_offset;
	std::vector<double> cost; // Cost on each interval (for mutation)
    bool param_solved; // Whether the knots, control points and costs have been computed
	bool knot_solved;

	MyPairSolution()
	{
		param_solved = false;
		knot_solved = false;
		crv.degree = 3;
	}

	std::string to_string() const
	{
		std::ostringstream out;
		out<<"{";
		/*for(int i=0;i<param.size();i++)
			out<<(i?",":"")<<std::setprecision(10)<<param[i];
		out<<"}, ";
        out<<"{";
		for(int i=0;i<knot_offset.size();i++)
			out<<(i?",":"")<<std::setprecision(10)<<knot_offset[i];*/
		out<<"}, ";
		return out.str();
	}
};

struct MyPairMiddleCost
{
    double cost_avg; // Average cost on the whole curve (for selection)
};

typedef EA::Genetic<MyPairSolution, MyPairMiddleCost> GA_Pair_Type;
typedef EA::GenerationType<MyPairSolution,MyPairMiddleCost> Gen_Pair_Type;

void init_pair_genes_tpl(
	MyPairSolution& p,
	const std::function<double(void)>& rnd01,
	const std::vector<glm::dvec3>& pts);

void offset2Knots(const std::vector<double>& param,
	const std::vector<double>& knot_offset,
	std::vector<double>& knot,
	int degree);

void print_res(std::vector<double>& param, std::vector<double>& knot_offset);

void pairCostComputation(
	const std::vector<glm::dvec3>& pts,
	MyPairSolution& p);

void paramCostComputation(
	const std::vector<glm::dvec3>& pts,
	const std::vector<double>& knot_offset,
	MyPairSolution& p);

void knotCostComputation(
	const std::vector<glm::dvec3>& pts,
	const std::vector<double>& param,
	MyPairSolution& p);

bool eval_pair_solution_tpl(
	MyPairSolution& p,
	MyPairMiddleCost& c,
	const std::vector<glm::dvec3>& pts);

bool eval_param_solution_tpl(
	MyPairSolution& p,
	MyPairMiddleCost& c,
	const std::vector<glm::dvec3>& pts,
	const std::vector<double>& knot_offset);

bool eval_knot_solution_tpl(
	MyPairSolution& p,
	MyPairMiddleCost& c,
	const std::vector<glm::dvec3>& pts,
	const std::vector<double>& param);

MyPairSolution mutate_param_tpl(
	MyPairSolution& X_base,
	const std::function<double(void)>& rnd01,
	double shrink_scale,
	const std::vector<glm::dvec3>& pts,
    const std::vector<double>& knot_offset);

MyPairSolution mutate_knot(
    MyPairSolution& X_base,
    const std::function<double(void)>& rnd01,
    double shrink_scale);

MyPairSolution crossover_param(
	const MyPairSolution& X1,
	const MyPairSolution& X2,
	const std::function<double(void)> &rnd01);

MyPairSolution crossover_knot(
    const MyPairSolution& X1,
    const MyPairSolution& X2,
    const std::function<double(void)> &rnd01);

void heuristicPairInit(
	const std::vector<glm::dvec3>& pts,
	std::vector<MyPairSolution>& user_initial_solutions,
	std::vector<double>& heuristic_cost);

double calculate_SO_total_fitness(const GA_Pair_Type::thisChromosomeType& X);

void SO_report_generation(
	int generation_number,
	const EA::GenerationType<MyPairSolution, MyPairMiddleCost>& last_generation,
	const MyPairSolution& best_genes);

struct PairResult
{
	double duration;
	double cost;
	std::string title;
	std::vector<double> param;
    std::vector<double> knot_offset;

	PairResult() : duration(0.0), cost(0.0), title(""), param() {}

	PairResult(double duration, double cost, std::string title, std::vector<double> param, std::vector<double> knot_offset):
		duration(duration),
		cost(cost),
		title(title),
		param(param),
        knot_offset(knot_offset)
	{
	}
};

struct OptRecord
{
	double avg_cost;
	double best_cost;
	OptRecord() : avg_cost(REAL_INF), best_cost(REAL_INF) {}
	OptRecord(double avg_cost, double best_cost):
		avg_cost(avg_cost),
		best_cost(best_cost)
	{
	}
};


void run_pair_test(
	bool multi_threading,
	bool dynamic_threading,
	int idle_delay_us,
	const std::vector<glm::dvec3>& pts,
	std::string title,
	PairResult& res,
	std::string& dir_save,
	std::string dir_output = "",
    int max_epoch = 100,
	double epsilon = std::numeric_limits<double>::epsilon());

void run_knot_test(
	bool multi_threading,
	bool dynamic_threading,
	int idle_delay_us,
	const std::vector<glm::dvec3>& pts,
	std::string title,
	const std::string path_output,
	int max_epoch,
	const std::vector<double>& param,
	double epsilon);

void batch_labeling(const std::string dir_data,
	const std::string dir_output,
    int max_epoch = 100,
	double epsilon = std::numeric_limits<double>::epsilon());
#endif
