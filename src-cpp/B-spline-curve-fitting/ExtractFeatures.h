#pragma once
#include <glm/glm.hpp>
#include <math.h>
#include "tinynurbs.h"
#include <iostream>
#include <vector>

/*计算每一段折线长度相对于最长折线的比例*/
void CalculateLineLengthRatio(std::vector<glm::dvec3>& pts, std::vector<double>& result) {

	result.resize(pts.size());

	result[0] = 0;
	double sumlen = 0;

	double maxlen;
	maxlen = sqrt(pow(pts[1].x - pts[0].x, 2) + pow(pts[1].y - pts[0].y, 2));

	for (int i = 1; i < result.size(); i++)
	{
		double len1 = sqrt(pow(pts[i].x - pts[i - 1].x, 2) + pow(pts[i].y - pts[i - 1].y, 2));
		sumlen += len1;

		if (len1 > maxlen)
			maxlen = len1;

	}

	for (int i = 1; i < result.size(); i++)
	{
		double len1 = sqrt(pow(pts[i].x - pts[i - 1].x, 2) + pow(pts[i].y - pts[i - 1].y, 2));
		result[i] = len1 / maxlen;
	}
}


/*计算每一个数据点处的角度变化：顺时针为正 逆时针为负*/ 
void CalculateLineAngle(std::vector<glm::dvec3>& pts, std::vector<double>& result) {

	result.resize(pts.size());

	result[0] = 0;
	result[result.size() - 1] = 0;

	double k, b;

	for (int i = 1; i < result.size() - 1; i++)
	{
		glm::dvec2 n1 = glm::dvec2(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);

		glm::dvec2 n2 = glm::dvec2(pts[i + 1].x - pts[i].x, pts[i + 1].y - pts[i].y);

		k = (pts[i - 1].y - pts[i].y) / (pts[i - 1].x - pts[i].x);

		b = (pts[i - 1].y * pts[i].x - pts[i].y * pts[i - 1].x) / (pts[i].x - pts[i - 1].x);

		double angle = acos((n1.x * n2.x + n1.y * n2.y) /
			sqrt(pow(n1.x, 2) + pow(n1.y, 2))
			/ sqrt(pow(n2.x, 2) + pow(n2.y, 2)));

		int flag = 1;
		if (k * pts[i + 1].x + b < pts[i + 1].y) {
			flag = -1;
		}
		result[i] = flag * angle;

	}

	/*
	std::cout << "Angle: ";
	for (int i = 0; i < result.size(); i++)
	{
		std::cout << result[i] << " ";
	}
	std::cout << std::endl;
	*/

}

