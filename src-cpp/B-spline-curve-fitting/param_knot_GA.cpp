#include "param_knot_GA.hpp"
#include "numeric"
#include <format>
#include "vec_file.hpp"

void avg_offset_init(MyPairSolution& p){
    AvgKnots(p.crv.degree, p.crv.knots, p.param);
	double num = p.param.size() - p.crv.degree - 1;
    p.knot_offset.resize(num);
    for (int i = 0; i < num; i++){
        double offset_i = (p.crv.knots[i+p.crv.degree+1] - p.param[i+1]) / (p.param[i+p.crv.degree] - p.param[i+1]);
        p.knot_offset[i] = offset_i;
    }
}

void init_pair_genes_tpl(
	MyPairSolution& p,
	const std::function<double(void)>& rnd01,
	const std::vector<glm::dvec3>& pts)
{
	p.param.resize(pts.size());
	p.param[0] = 0;
	p.param[pts.size() - 1] = 1;
	for (int i = 1; i < pts.size() - 1; i++)
	{
		p.param[i] = rnd01();
	}
	std::sort(p.param.begin(), p.param.end());
	avg_offset_init(p);
}

bool eval_pair_solution_tpl(
	MyPairSolution& p,
	MyPairMiddleCost& c,
	const std::vector<glm::dvec3>& pts)
{
	if (!p.param_solved || !p.knot_solved)
	{
		pairCostComputation(pts, p);
	}
	// Computing the average cost
	c.cost_avg = 0;
	for (int i = 0; i < p.cost.size(); i++)
	{
		c.cost_avg += p.cost[i];
	}
	c.cost_avg /= (p.param.size() - 1);
	return true; // genes are accepted
}

bool eval_param_solution_tpl(
	MyPairSolution& p,
	MyPairMiddleCost& c,
	const std::vector<glm::dvec3>& pts,
	const std::vector<double>& knot_offset)
{
	if (!p.param_solved)
	{
		paramCostComputation(pts, knot_offset, p);
	}
	// Computing the average cost
	c.cost_avg = 0;
	for (int i = 0; i < p.cost.size(); i++)
	{
		c.cost_avg += p.cost[i];
	}
	c.cost_avg /= (p.param.size() - 1);
	return true; // genes are accepted
}

bool eval_knot_solution_tpl(
	MyPairSolution& p,
	MyPairMiddleCost& c,
	const std::vector<glm::dvec3>& pts,
	const std::vector<double>& param)
{
	if (!p.knot_solved)
	{
		knotCostComputation(pts, param, p);
	}
	// Computing the average cost
	c.cost_avg = 0;
	for (int i = 0; i < p.cost.size(); i++)
	{
		c.cost_avg += p.cost[i];
	}
	c.cost_avg /= (param.size() - 1);
	return true; // genes are accepted
}

double calculate_SO_total_fitness(const GA_Pair_Type::thisChromosomeType& X)
{
	// finalize the cost
	if (X.middle_costs.cost_avg < 1e-4) {
		return INFINITY;
	}
	return X.middle_costs.cost_avg;
}

double calculate_param_fitness_tpl(GA_Pair_Type::thisChromosomeType& X,
	const std::vector<glm::dvec3>& pts,
	const std::vector<double>& knot_offset)
{
	if (X.genes.param_solved) {
		X.genes.param_solved = false;
		X.genes.knot_solved = false;
		if (X.middle_costs.cost_avg < 1e-4) {
			return INFINITY;
		}
		return X.middle_costs.cost_avg;
	}
	MyPairSolution p = X.genes;
	MyPairMiddleCost c;
	eval_param_solution_tpl(p, c, pts, knot_offset);
	if (c.cost_avg < 1e-4) {
		return INFINITY;
	}
	return c.cost_avg;
}

double calculate_knot_fitness_tpl(GA_Pair_Type::thisChromosomeType& X,
	const std::vector<glm::dvec3>& pts,
	const std::vector<double>& param)
{
	if (X.genes.knot_solved) {
		X.genes.param_solved = false;
		X.genes.knot_solved = false;
		if (X.middle_costs.cost_avg < 1e-4) {
			return INFINITY;
		}
		return X.middle_costs.cost_avg;
	}
	MyPairSolution p = X.genes;
	MyPairMiddleCost c;
	eval_knot_solution_tpl(p, c, pts, param);
	if (c.cost_avg < 1e-4) {
		return INFINITY;
	}
	return c.cost_avg;
}

void SO_report_generation(
	int generation_number,
	const EA::GenerationType<MyPairSolution, MyPairMiddleCost>& last_generation,
	const MyPairSolution& best_genes)
{
	std::cout
		<< "Generation [" << generation_number << "], "
		<< "Best=" << last_generation.best_total_cost << ", "
		<< "Average=" << last_generation.average_cost << ", "
		<< "Best genes=(" << best_genes.to_string() << ")" << ", "
		<< "Exe_time=" << last_generation.exe_time
		<< std::endl;
}

void offset2Knots(const std::vector<double>& param,
	const std::vector<double>& knot_offset,
	std::vector<double>& knot,
	int degree)
{
	knot.resize(knot_offset.size() + 2 * degree + 2);
	for (int i = 0; i < degree + 1; i++)
	{
		knot[i] = 0;
		knot[knot.size() - 1 - i] = 1;
	}
	for (int i = 0; i < knot_offset.size(); i++) {
		knot[i + degree + 1] = param[i + 1] + knot_offset[i] * (param[i + degree] - param[i + 1]);
	}
}

void pairCostComputation(
	const std::vector<glm::dvec3>& pts,
	MyPairSolution& p)
{
    // Computing knots
	offset2Knots(p.param, p.knot_offset, p.crv.knots, p.crv.degree);
	// Computing control points corresponding to the parameterization
	GlobalInterp(pts, 3, p.crv.knots, p.param, p.crv.control_points);
	// Computing the product of arc height and maximum curvature on each interval 
	p.cost.resize(p.param.size() - 1);
	for (int i = 1; i < p.param.size(); i++)
	{
		p.cost[i - 1] = calculate2013Criteria(p.crv, p.param[i - 1], p.param[i]);
		//p.cost[i - 1] = calculateDerVariance(p.crv, p.param[i - 1], p.param[i]);
		//p.cost[i - 1] = calculateModifiedCriteria(p.crv, p.param[i - 1], p.param[i]);
	}
	//p.param_solved = true;
}

void paramCostComputation(
	const std::vector<glm::dvec3>& pts,
	const std::vector<double>& knot_offset,
	MyPairSolution& p)
{
    // Computing knots
	offset2Knots(p.param, knot_offset, p.crv.knots, p.crv.degree);
	// Computing control points corresponding to the parameterization
	GlobalInterp(pts, 3, p.crv.knots, p.param, p.crv.control_points);
	// Computing the product of arc height and maximum curvature on each interval 
	p.cost.resize(p.param.size() - 1);
	for (int i = 1; i < p.param.size(); i++)
	{
		p.cost[i - 1] = calculate2013Criteria(p.crv, p.param[i - 1], p.param[i]);
		//p.cost[i - 1] = calculateDerVariance(p.crv, p.param[i - 1], p.param[i]);
		//p.cost[i - 1] = calculateModifiedCriteria(p.crv, p.param[i - 1], p.param[i]);
	}
	p.param_solved = true;
	p.knot_solved = false;
}

void knotCostComputation(
	const std::vector<glm::dvec3>& pts,
	const std::vector<double>& param,
	MyPairSolution& p)
{
    // Computing knots
	offset2Knots(param, p.knot_offset, p.crv.knots, p.crv.degree);
	// Computing control points corresponding to the parameterization
	GlobalInterp(pts, 3, p.crv.knots, param, p.crv.control_points);
	// Computing the product of arc height and maximum curvature on each interval 
	p.cost.resize(param.size() - 1);
	for (int i = 1; i < param.size(); i++)
	{
		p.cost[i - 1] = calculate2013Criteria(p.crv, param[i - 1], param[i]);
		//p.cost[i - 1] = calculateDerVariance(p.crv, param[i - 1], param[i]);
		//p.cost[i - 1] = calculateModifiedCriteria(p.crv, param[i - 1], param[i]);
	}
	p.knot_solved = true;
	p.param_solved = false;
}

MyPairSolution mutate_tpl(
	MyPairSolution& X_base,
	const std::function<double(void)> &rnd01,
	double shrink_scale,
	const std::vector<glm::dvec3>& pts)
{
	if (!X_base.param_solved || !X_base.knot_solved)
	{
		pairCostComputation(pts, X_base);
	}
	MyPairSolution X_new;
	// Finding the part with the worst error
	double cost_max = 0, cost_tmp;
	int idx_max = 1;
	for (int i = 1; i < X_base.cost.size(); i++)
	{
		cost_tmp = X_base.cost[i - 1] + X_base.cost[i];
		if (cost_tmp > cost_max){
			cost_max = cost_tmp;
			idx_max = i;
		}
	}
	// Mutation on the part with the worst error
	X_new.param.assign(X_base.param.begin(), X_base.param.end());
    X_new.knot_offset.assign(X_base.knot_offset.begin(), X_base.knot_offset.end());
	double r = rnd01();
	X_new.param[idx_max] = X_new.param[idx_max - 1] + 0.5 * (1 + (2 * r - 1) * shrink_scale * 0.6) * (X_new.param[idx_max + 1] - X_new.param[idx_max - 1]);
	return X_new;
}

MyPairSolution mutate_param_tpl(
	MyPairSolution& X_base,
	const std::function<double(void)> &rnd01,
	double shrink_scale,
	const std::vector<glm::dvec3>& pts,
    const std::vector<double>& knot_offset)
{
	if (!X_base.param_solved)
	{
		paramCostComputation(pts, knot_offset, X_base);
	}
	MyPairSolution X_new;
	// Finding the part with the worst error
	double cost_max = 0, cost_tmp;
	int idx_max = 1;
	for (int i = 1; i < X_base.cost.size(); i++)
	{
		cost_tmp = X_base.cost[i - 1] + X_base.cost[i];
		if (cost_tmp > cost_max){
			cost_max = cost_tmp;
			idx_max = i;
		}
	}
	// Mutation on the part with the worst error
	X_new.param.assign(X_base.param.begin(), X_base.param.end());
    X_new.knot_offset.assign(X_base.knot_offset.begin(), X_base.knot_offset.end());
	double r = rnd01();
	X_new.param[idx_max] = X_new.param[idx_max - 1] + 0.5 * (1 + (2 * r - 1) * shrink_scale * 0.8) * (X_new.param[idx_max + 1] - X_new.param[idx_max - 1]);
	return X_new;
}

MyPairSolution mutate_knot(
    MyPairSolution& X_base,
    const std::function<double(void)>& rnd01,
    double shrink_scale)
{
    // Randomly choose the part to mutate
    MyPairSolution X_new;
    int num = X_base.knot_offset.size();
    int idx = floor(rnd01() * num);
    // Mutation
    X_new.param.assign(X_base.param.begin(), X_base.param.end());
    X_new.knot_offset.assign(X_base.knot_offset.begin(), X_base.knot_offset.end());
    double r = rnd01();
    X_new.knot_offset[idx] = 0.5 * (1 + (2 * r - 1) * shrink_scale * 0.8);
    return X_new;
}

MyPairSolution crossover_param(
	const MyPairSolution& X1,
	const MyPairSolution& X2,
	const std::function<double(void)> &rnd01)
{
	MyPairSolution X_new;
	// Randomly choosing a crossover point
    int idx = floor(rnd01() * (X1.param.size() - 2)) + 1;
	// Choosing between X1 and X2
	double choice = rnd01();
	if (choice < 0.5)
	{
		X_new.param.assign(X1.param.begin(), X1.param.end());
        X_new.knot_offset.assign(X1.knot_offset.begin(), X1.knot_offset.end());
		X_new.param[idx] = X2.param[idx];
	}
	else{
		X_new.param.assign(X2.param.begin(), X2.param.end());
        X_new.knot_offset.assign(X2.knot_offset.begin(), X2.knot_offset.end());
		X_new.param[idx] = X1.param[idx];
	}
	std::sort(X_new.param.begin(), X_new.param.end());
	return X_new;
}

MyPairSolution crossover_knot(
    const MyPairSolution& X1,
    const MyPairSolution& X2,
    const std::function<double(void)> &rnd01)
{
    // Randomly choosing a crossover point
	int idx = floor(rnd01() * (X1.knot_offset.size()));
    // Choosing between X1 and X2
    double choice = rnd01();
    MyPairSolution X_new;
    if (choice < 0.5){
        X_new.param.assign(X1.param.begin(), X1.param.end());
        X_new.knot_offset.assign(X1.knot_offset.begin(), X1.knot_offset.end());
        X_new.knot_offset[idx] = X2.knot_offset[idx];
    }
    else{
        X_new.param.assign(X2.param.begin(), X2.param.end());
        X_new.knot_offset.assign(X2.knot_offset.begin(), X2.knot_offset.end());
        X_new.knot_offset[idx] = X1.knot_offset[idx];
    }
    return X_new;
}

void heuristicPairInit(
	const std::vector<glm::dvec3>& pts,
	std::vector<MyPairSolution>& user_initial_solutions,
	std::vector<double>& heuristic_cost)
{
	// Generate initial population with heuristic methods
	user_initial_solutions.clear();
	heuristic_cost.clear();
	MyPairMiddleCost tmp_cost;
	// Uniform
	MyPairSolution geneUniform;
	UniformInterp(pts, 3, geneUniform.crv.knots, geneUniform.param);
    avg_offset_init(geneUniform);
	eval_pair_solution_tpl(geneUniform, tmp_cost, pts);
	user_initial_solutions.push_back(geneUniform);
	heuristic_cost.push_back(tmp_cost.cost_avg);
	// Chord length
	MyPairSolution geneChord;
	ChordInterp(pts, 3, geneChord.crv.knots, geneChord.param);
    avg_offset_init(geneChord);
	eval_pair_solution_tpl(geneChord, tmp_cost, pts);
	user_initial_solutions.push_back(geneChord);
	heuristic_cost.push_back(tmp_cost.cost_avg);
	// Centripetal
	MyPairSolution geneCentripetal;
	CentripetalInterp(pts, 3, geneCentripetal.crv.knots, geneCentripetal.param);
    avg_offset_init(geneCentripetal);
	eval_pair_solution_tpl(geneCentripetal, tmp_cost, pts);
	user_initial_solutions.push_back(geneCentripetal);
	heuristic_cost.push_back(tmp_cost.cost_avg);
	// Universal
	MyPairSolution geneUniversal;
	UniversalInterp(pts, 3, geneUniversal.crv.knots, geneUniversal.param);
    avg_offset_init(geneUniversal);
	eval_pair_solution_tpl(geneUniversal, tmp_cost, pts);
	user_initial_solutions.push_back(geneUniversal);
	heuristic_cost.push_back(tmp_cost.cost_avg);
	// Foley
	MyPairSolution geneFoley;
	CorrectChordInterp(pts, 3, geneFoley.crv.knots, geneFoley.param);
    avg_offset_init(geneFoley);
	eval_pair_solution_tpl(geneFoley, tmp_cost, pts);
	user_initial_solutions.push_back(geneFoley);
	heuristic_cost.push_back(tmp_cost.cost_avg);
	// Fang
	MyPairSolution geneFang;
	RefinedCentripetalInterp(pts, 3, geneFang.crv.knots, geneFang.param);
    avg_offset_init(geneFang);
	eval_pair_solution_tpl(geneFang, tmp_cost, pts);
	user_initial_solutions.push_back(geneFang);
	heuristic_cost.push_back(tmp_cost.cost_avg);
	// Modified chord length
	MyPairSolution geneModifiedChord;
	ModifiedChordInterp(pts, 3, geneModifiedChord.crv.knots, geneModifiedChord.param);
    avg_offset_init(geneModifiedChord);
	eval_pair_solution_tpl(geneModifiedChord, tmp_cost, pts);
	user_initial_solutions.push_back(geneModifiedChord);
	heuristic_cost.push_back(tmp_cost.cost_avg);
	// ZCM
	MyPairSolution geneZCM;
	ZCMInterp(pts, 3, geneZCM.crv.knots, geneZCM.param);
    avg_offset_init(geneZCM);
	eval_pair_solution_tpl(geneZCM, tmp_cost, pts);
	user_initial_solutions.push_back(geneZCM);
	heuristic_cost.push_back(tmp_cost.cost_avg);
}

std::vector<PairResult> test_results;

bool real_vec_equal(const std::vector<double>& v1, const std::vector<double>& v2) {
	if (v1.size() != v2.size()) return false;
	for (int i = 0; i < v1.size(); i++) {
		if (std::abs(v1[i] - v2[i]) > 1e-6) return false;
	}
	return true;
}

void print_res(std::vector<double>& param, std::vector<double>& knot_offset) {
	cout << "param: {";
	for (int i = 0; i < param.size(); i++) {
		std::cout << param[i] << ", ";
	}
	cout << "}, ";
	cout << "knot_offset: {";
	for (int i = 0; i < knot_offset.size(); i++) {
		std::cout << knot_offset[i] << ", ";
	}
	cout << "}" << endl;
}

void run_pair_test(
	bool multi_threading,
	bool dynamic_threading,
	int idle_delay_us,
	const std::vector<glm::dvec3>& pts,
	std::string title,
	PairResult& res,
	std::string& dir_save,
	const std::string dir_output,
    int max_epoch,
	double epsilon)
{
	std::cout << "-------------------------------" << std::endl;
	std::cout << "Running the test: " << title << std::endl;

	EA::Chronometer timer;
	timer.tic();
    function<void(MyPairSolution&,const function<double(void)> &rnd01)> init_genes = [&](
		MyPairSolution& p,
		const std::function<double(void)>& rnd01)
	{
        init_pair_genes_tpl(p, rnd01, pts);
    };
	function<bool(MyPairSolution&,MyPairMiddleCost&)> eval_solution = [&](
		MyPairSolution& p,
		MyPairMiddleCost &c)
	{
        return eval_pair_solution_tpl(p, c, pts);
    };
	function<MyPairSolution(MyPairSolution&, const function<double(void)>& rnd01, double shrink_scale)> mutate = [&](
		MyPairSolution& p,
		const std::function<double(void)>& rnd01,
		double shrink_scale
		)
	{
		return mutate_tpl(p, rnd01, shrink_scale, pts);
	};
	GA_Pair_Type param_obj;
	param_obj.problem_mode = EA::GA_MODE::SOGA;
	param_obj.multi_threading = multi_threading;
	param_obj.dynamic_threading = dynamic_threading;
	param_obj.idle_delay_us = idle_delay_us;
	param_obj.verbose = false;
	param_obj.population = 25;
	// Initializing the parameter population
	std::vector<MyPairSolution> user_initial_solutions;
	std::vector<double> heuristic_cost;
	heuristicPairInit(pts, user_initial_solutions, heuristic_cost);
	string method_name[8] = {"uniform", "chord", "centripetal", "universal",
		"foley", "fang", "modified_chord", "zcm"};
	auto min_it = std::min_element(heuristic_cost.begin(), heuristic_cost.end());
	int min_idx = std::distance(heuristic_cost.begin(), min_it);
	param_obj.user_initial_solutions = user_initial_solutions;
	param_obj.generation_max = max_epoch;
	param_obj.calculate_SO_total_fitness = calculate_SO_total_fitness;
	param_obj.init_genes = init_genes;
	param_obj.eval_solution = eval_solution;
	param_obj.mutate = mutate;
	param_obj.crossover = crossover_param;
	param_obj.SO_report_generation = SO_report_generation;
	param_obj.best_stall_max = 10; // recommended value is 10
	param_obj.average_stall_max = 20; // tentative
	param_obj.tol_stall_best = 1e-6; // tentative
	param_obj.tol_stall_average = 1e-6; // tentative
	param_obj.elite_count = 15; // tentative
	param_obj.crossover_fraction = 1.0; // Including crossover and mutation
	param_obj.mutation_rate = 0.1;
	std::cout<<"Initializing ..."<<std::endl;
	param_obj.solve_init();
	param_obj.solve_next_generation();
	Gen_Pair_Type& param_last_gen = param_obj.last_generation;
	/*best_param.assign(param_last_gen.chromosomes[param_last_gen.best_chromosome_index].genes.param.begin(),
		param_last_gen.chromosomes[param_last_gen.best_chromosome_index].genes.param.end());*/
	// Preparing for later generations
	param_obj.crossover = crossover_param;
	// Initializing the knot offset population
	GA_Pair_Type knot_obj = param_obj;
	knot_obj.mutate = mutate_knot;
	knot_obj.crossover = crossover_knot;
	// last generation
	Gen_Pair_Type& knot_last_gen = knot_obj.last_generation;
	std::vector<double>& best_param = param_last_gen.chromosomes[param_last_gen.best_chromosome_index].genes.param, 
		& best_knot_offset = knot_last_gen.chromosomes[knot_last_gen.best_chromosome_index].genes.knot_offset;
	double last_cost = REAL_INF, crt_cost = REAL_INF;
	int stall_cnt = 0;
    for (int i = 0; i < max_epoch; i++){
		// Optimize the knot vector
		if (best_knot_offset.size()) {
			std::cout << "Optimizing the knot vector" << std::endl;
			function<bool(MyPairSolution&, MyPairMiddleCost&)> eval_knot_solution = [&](
				MyPairSolution& p,
				MyPairMiddleCost& c)
			{
				return eval_knot_solution_tpl(p, c, pts, best_param);
			};
			function<double(GA_Pair_Type::thisChromosomeType&)> calculate_knot_fitness = [&](
				GA_Pair_Type::thisChromosomeType& X)
			{
				return calculate_knot_fitness_tpl(X, pts, best_param);
			};
			knot_obj.eval_solution = eval_knot_solution;
			knot_obj.calculate_SO_total_fitness = calculate_knot_fitness;
			knot_obj.solve_next_generation();
			/*best_knot_offset.assign(knot_last_gen.chromosomes[knot_last_gen.best_chromosome_index].genes.knot_offset.begin(),
				knot_last_gen.chromosomes[knot_last_gen.best_chromosome_index].genes.knot_offset.end());*/
		}
		// Optimize the parameters
		std::cout<<"Optimizing the parameters"<<std::endl;
		function<bool(MyPairSolution&,MyPairMiddleCost&)> eval_param_solution = [&](
			MyPairSolution& p,
			MyPairMiddleCost &c)
		{
			return eval_param_solution_tpl(p, c, pts, best_knot_offset);
		};
		function<MyPairSolution(MyPairSolution&, const function<double(void)>& rnd01, double shrink_scale)> mutate_param = [&](
			MyPairSolution& p,
			const std::function<double(void)>& rnd01,
			double shrink_scale
			)
		{
			return mutate_param_tpl(p, rnd01, shrink_scale, pts, best_knot_offset);
		};
		function<double(GA_Pair_Type::thisChromosomeType&)> calculate_param_fitness = [&](
			GA_Pair_Type::thisChromosomeType& X)
		{
			return calculate_param_fitness_tpl(X, pts, best_knot_offset);
		};
		param_obj.mutate = mutate_param;
		param_obj.eval_solution = eval_param_solution;
		param_obj.calculate_SO_total_fitness = calculate_param_fitness;
		param_obj.solve_next_generation();
		/*best_param.assign(param_last_gen.chromosomes[param_last_gen.best_chromosome_index].genes.param.begin(),
			param_last_gen.chromosomes[param_last_gen.best_chromosome_index].genes.param.end());*/
		//print_res(best_param, best_knot_offset);
		crt_cost = param_last_gen.chromosomes[param_last_gen.best_chromosome_index].total_cost;
		if (std::abs(last_cost - crt_cost) < epsilon){
			stall_cnt++;
		}
		else{
			stall_cnt=0;
		}
		if (stall_cnt > param_obj.best_stall_max) break;
		last_cost = crt_cost;
    }
	
	double duration = timer.toc();
	std::cout<<"The problem is optimized in "<<duration<<" seconds."<<std::endl;
	cout << "best record:" << param_obj.best_record.back() << ", "<< knot_obj.best_record.back() << endl;
	res = PairResult(duration, param_last_gen.chromosomes[param_last_gen.best_chromosome_index].total_cost,
	                method_name[min_idx], best_param, best_knot_offset);
	if (dir_output != ""){
		dir_save = dir_output + "/" + method_name[min_idx] + "/" + title;
		writeVectorToFile(heuristic_cost, dir_save + "/heuristic_cost.bin");
		//writeVectorToFile(param_obj.avg_record, dir_save + "/avg_cost.bin");
		writeVectorToFile(param_obj.best_record, dir_save + "/best_cost.bin");
		writeVectorToFile(best_param, dir_save + "/param.bin");
		writeVectorToFile(best_knot_offset, dir_save + "/knot_offset.bin");
		std::ofstream data_file(dir_save + "/point_data.txt");
		data_file << pts.size() <<std::endl;
		for (int i = 0; i < pts.size(); i++) {
			data_file << pts[i].x << " " << pts[i].y << std::endl;
		}
		data_file.close();
	}
}

void run_knot_test(
	bool multi_threading,
	bool dynamic_threading,
	int idle_delay_us,
	const std::vector<glm::dvec3>& pts,
	std::string title,
	const std::string path_output,
	int max_epoch,
	const std::vector<double>& param,
	double epsilon)
{
	std::cout << "-------------------------------" << std::endl;
	std::cout << "Running the test: " << title << std::endl;

	EA::Chronometer timer;
	timer.tic();
	function<void(MyPairSolution&, const function<double(void)>& rnd01)> init_genes = [&](
		MyPairSolution& p,
		const std::function<double(void)>& rnd01)
		{
			init_pair_genes_tpl(p, rnd01, pts);
		};
	function<bool(MyPairSolution&, MyPairMiddleCost&)> eval_solution = [&](
		MyPairSolution& p,
		MyPairMiddleCost& c)
		{
			return eval_pair_solution_tpl(p, c, pts);
		};
	function<MyPairSolution(MyPairSolution&, const function<double(void)>& rnd01, double shrink_scale)> mutate = [&](
		MyPairSolution& p,
		const std::function<double(void)>& rnd01,
		double shrink_scale
		)
		{
			return mutate_tpl(p, rnd01, shrink_scale, pts);
		};
	GA_Pair_Type knot_obj;
	knot_obj.problem_mode = EA::GA_MODE::SOGA;
	knot_obj.multi_threading = multi_threading;
	knot_obj.dynamic_threading = dynamic_threading;
	knot_obj.idle_delay_us = idle_delay_us;
	knot_obj.verbose = false;
	knot_obj.population = 25;
	// Initializing the population
	std::vector<MyPairSolution> user_initial_solutions;
	std::vector<double> heuristic_cost;
	heuristicPairInit(pts, user_initial_solutions, heuristic_cost);
	string method_name[8] = { "uniform", "chord", "centripetal", "universal",
		"foley", "fang", "modified_chord", "zcm" };
	auto min_it = std::min_element(heuristic_cost.begin(), heuristic_cost.end());
	int min_idx = std::distance(heuristic_cost.begin(), min_it);
	knot_obj.user_initial_solutions = user_initial_solutions;
	knot_obj.generation_max = max_epoch;
	function<bool(MyPairSolution&, MyPairMiddleCost&)> eval_knot_solution = [&](
		MyPairSolution& p,
		MyPairMiddleCost& c)
		{
			return eval_knot_solution_tpl(p, c, pts, param);
		};
	function<double(GA_Pair_Type::thisChromosomeType&)> calculate_knot_fitness = [&](
		GA_Pair_Type::thisChromosomeType& X)
		{
			return calculate_knot_fitness_tpl(X, pts, param);
		};
	knot_obj.calculate_SO_total_fitness = calculate_knot_fitness;
	knot_obj.init_genes = init_genes;
	knot_obj.eval_solution = eval_knot_solution;
	knot_obj.mutate = mutate_knot;
	knot_obj.crossover = crossover_knot;
	knot_obj.SO_report_generation = SO_report_generation;
	knot_obj.best_stall_max = 10; // recommended value is 10
	knot_obj.average_stall_max = 20; // tentative
	knot_obj.tol_stall_best = epsilon; // tentative
	knot_obj.tol_stall_average = epsilon; // tentative
	knot_obj.elite_count = 15; // tentative
	knot_obj.crossover_fraction = 1.0; // Including crossover and mutation
	knot_obj.mutation_rate = 0.1;
	knot_obj.solve();
	double duration = timer.toc();
	std::cout << "The problem is optimized in " << duration << " seconds." << std::endl;
	Gen_Pair_Type last_gen = knot_obj.last_generation;
	const std::vector<double>& best_knot_offset = last_gen.chromosomes[last_gen.best_chromosome_index].genes.knot_offset;
	std::vector<double> best_knot;
	offset2Knots(param, best_knot_offset, best_knot, 3);
	if (path_output != "") {
		writeVectorToFile(best_knot, path_output);
	}
}

void batch_labeling(const std::string dir_data,
	const std::string dir_output,
    int max_epoch,
	double epsilon)
{
	WIN32_FIND_DATAA findFileData;
	std::string dir_tpl = dir_data + "/*.txt";
    HANDLE hFind = FindFirstFileA(dir_tpl.c_str(), &findFileData);
    
    if (hFind == INVALID_HANDLE_VALUE) {
        std::cerr << "No files found in the directory: " << dir_data << std::endl;
        return;
    }
	do {
		std::string filePath = dir_data + "/" + findFileData.cFileName;
		std::ifstream file(filePath);
		std::string recordDir = dir_output + "/record";
		create_directory_if_not_exists(recordDir);
		if (file.is_open()) {
			size_t sep_pos = filePath.find_last_of("\\/");
			std::string name = (sep_pos != std::string::npos) ? filePath.substr(sep_pos + 1) : filePath;
			name.erase(name.size() - 4);
			int id;
			try {
				id = std::stoi(name);
			}
			catch (const std::invalid_argument& e) {
				std::cerr << "Invalid argument: " << e.what() << " " << name << std::endl;
				continue;
			}
			std::string recordPath = recordDir + "/" + name + ".txt";
			// Check if it's already labeled
			std::ifstream recordFileIn(recordPath);
			if (recordFileIn.good()) continue;
			std::cout << "Reading file: " << filePath << std::endl;
			std::vector<glm::dvec3> pts;
			int fitting_num;
			file >> fitting_num;
			for (int i = 0; i < fitting_num; i++)
			{
				double x, y;
				file >> x >> y;
				pts.push_back(glm::dvec3(x, y, 0.0));
			}
			file.close();
			// Repeat the optimization for several times
			PairResult best_res, tmp_res;
			int best_idx;
			std::string dir_tmp;
			for (int i = 0; i < 5; i++){
				run_pair_test(true, true, 1000, pts, name + "/" + std::to_string(i), tmp_res, dir_tmp, dir_output, max_epoch, epsilon);
				if (i==0 || tmp_res.cost < best_res.cost){
					best_res = tmp_res;
					best_idx = i;
				}
			}
			// Save the best result
			std::cout << "Best: " << best_res.cost << std::endl;
			std::cout << "**************************************" << std::endl;
			std::string dir_save = dir_output + "/" + best_res.title + "/" + name;
			writeVectorToFile(best_res.param, dir_save + "/param.bin");
			std::ofstream cost_file(dir_save + "/cost.bin", std::ios::binary);
			cost_file.write(reinterpret_cast<const char*>(&(best_res.cost)), sizeof(best_res.cost));
			copyFile(dir_save + "/0/point_data.txt", dir_save + "/point_data.txt");
			std::ofstream recordFileOut(recordPath);
			if (!recordFileOut.is_open()) std::cerr << "Failed to open " << recordPath << std::endl;
			else {
				recordFileOut << dir_save << std::endl << best_idx << std::endl << best_res.cost;
				recordFileOut.close();
			}
		}
		else std::cerr << "Failed to open file: " << filePath << std::endl;
	} while (FindNextFileA(hFind, &findFileData) != 0);

	FindClose(hFind);
	
}
