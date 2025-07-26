#pragma once
#include <glm/glm.hpp>
#include <vector>

#define PI 3.1415926
#define PROJECT_ROOT "D:\\BSplineLearning"

void GlobalInterp(const std::vector<glm::dvec3>& points, int degree, const std::vector<double>& knots, 
					const std::vector<double>& param, std::vector<glm::dvec3>& controlPts);

void UniformInterp(const std::vector<glm::dvec3>& points, int degree, 
					std::vector<double>& knots, std::vector<double>& param);


void ChordInterp(const std::vector<glm::dvec3>& points, int degree, 
					std::vector<double>& knots, std::vector<double>& param);


void UniversalInterp(const std::vector<glm::dvec3>& points, int degree,
						std::vector<double>& knots, std::vector<double>& param);


void CentripetalInterp(const std::vector<glm::dvec3>& points, int degree,
						std::vector<double>& knots, std::vector<double>& param);


void CorrectChordInterp(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param); //�����ҳ���������


void RefinedCentripetalInterp(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param);  //2013�������л������Ĳ������������ĸĽ�����


void ClassfierInterp(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param, const std::vector<int>& minis);//������һ�κ����һ�γ�ֵ��Ȼ��������С���˷��������


void ClassfierInterp1(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param, std::vector<int> minis); //������ֵ��ֱ�ӷ��������

void ClassfierInterp2(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param, std::vector<int> minis); //������һ�γ�ֵ��ֱ�����ñ�����⣬��ʹ�÷��������


void ClassfierInterp3(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param, std::vector<int> minis); //������һ�γ�ֵ1��ֱ�����ñ�����⣻�������һ�γ�ֵ1��Ȼ���ñ�����⣻���ν����ƽ��ֵ

void ClassfierInterp4(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param, std::vector<int> minis); //������һ�γ�ֵ1��ֱ�����ñ�����⣻�������һ�γ�ֵ�ҳ�����1��Ȼ���ñ�����⣻���ν����ƽ��ֵ

void ModifiedChordInterp(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param);  //2022��������Բ�����ҳ��ķ���


void ZCMInterp(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param);  //ZCM����������

void RegressorInterp(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param, std::vector<double> intervals);

void VarRegressorInterp(const std::vector<glm::dvec3>& points, int degree,
	std::vector<double>& knots, std::vector<double>& param, std::vector<double> intervals, int local_len=4);

void GlobalInterp_Closed(std::vector<glm::dvec3> points, int degree, std::vector<double>& knots,
	std::vector<double>& param, std::vector<glm::dvec3>& controlPts);


void AvgKnots(int degree, std::vector<double>& knots,
	const std::vector<double>& param);

double paramKnotConditionNumber(int degree, const std::vector<double>& knots,
	const std::vector<double>& param);