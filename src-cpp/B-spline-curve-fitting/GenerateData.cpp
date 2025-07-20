#include"GenerateData.h"


std::string NumToStr(int num) {

	std::string name;
	if (num == 0)
	{
		name.push_back('0');
	}
	else {
		while (num > 0)
		{
			name.push_back('0' + num % 10);
			num /= 10;
		}
		for (int i = 0; i < name.length() / 2; i++)
		{
			char t = name[i];
			name[i] = name[name.length() - i - 1];
			name[name.length() - i - 1] = t;
		}
	}
	return name;
}

double TransferAngle(double angle) {

	
	if (angle <= 0) {
		return -angle;
	}
	else {
		return M_PI * 2 - angle;
	}
	
	/*
	if (angle < 0) {
		return M_PI*2 + angle;
	}
	*/
}


void GenerateDataDirect(int dataNum, int pNum, int flag_num, string DataDir) {

	glm::dvec3 currentdata = glm::dvec3(5, 5, 0);
	glm::dvec3 nextdata;
	glm::dvec3 lastdata;

	double dis, lastangle = 0, deltaangle, currentangle;
	int min = 2, max = 20;

	int m1 = 0;
	int m2 = 0;
	int m3 = 0;
	int m4 = 0;

	//srand((unsigned int)time(NULL));

	for (int k = 0; k < dataNum; k++)
	{

		//std::cout << k << std::endl;

		std::string name;
		int tmp = k;
		if (tmp == 0)
		{
			name.push_back('0');
		}
		else {
			while (tmp > 0)
			{
				name.push_back('0' + tmp % 10);
				tmp /= 10;
			}
			for (int i = 0; i < name.length() / 2; i++)
			{
				char t = name[i];
				name[i] = name[name.length() - i - 1];
				name[name.length() - i - 1] = t;
			}
		}

		std::ofstream outfile;
		outfile.open(DataDir + "/" + name + ".txt");
		outfile << pNum << std::endl;

		currentdata = glm::dvec3(5, 5, 0);

		int data_mode = k % 4 + 1;

		//int data_mode = k / (dataNum / 4) + 1;
		//if (flag_num <= 2)
		//	data_mode = 2;
		//int data_mode = 1;

		for (int i = 0; i < pNum; i++) {
			if (i != 0) {
				currentdata = nextdata;
			}
			outfile << currentdata.x << " " << currentdata.y << std::endl;
			//dataPoints[i][0] = currentdata.x;
			//dataPoints[i][1] = currentdata.y;
			//dataPoints[i][2] = 0;
			switch (data_mode) {
			case 1:
				//Abrupt Changing Angle Length
				dis = min + (max - min) * (rand() / double(RAND_MAX));
				if (m1 % 3 != 0)
					deltaangle = (-180 + (360 * rand() / (double(RAND_MAX)))) / 180 * M_PI_2;
				else
					deltaangle = (-180 + (360 * rand() / (double(RAND_MAX)))) / 180 * M_PI * 5 / 6;
				m1++;
				break;
			case 2:
				//NearEqualLengthLittleChangingAngle
				dis = 6;
				if (m2 % 3 != 0)
					deltaangle = (-180 + (360 * rand() / (double(RAND_MAX)))) / 180 * M_PI_2;
				else
					deltaangle = (-180 + (360 * rand() / (double(RAND_MAX)))) / 180 * M_PI_2 * 4 / 3;
				m2++;
				break;
			case 3:
				//AbruptChangingLengthLittleChangingAngle
				dis = min + (max - min) * (rand() / double(RAND_MAX));
				if (m3 % 3 != 0)
					deltaangle = (-180 + (360 * rand() / (double(RAND_MAX)))) / 180 * M_PI_2;
				else
					deltaangle = (-180 + (360 * rand() / (double(RAND_MAX)))) / 180 * M_PI_2 * 4 / 3;
				m3++;
				break;
			case 4:
				//NearEqualLengthAbruptChangingAngle
				dis = 6;
				if (m4 % 3 != 0)
					deltaangle = (-180 + (360 * rand() / (double(RAND_MAX)))) / 180 * M_PI_2;
				else
					deltaangle = (-180 + (360 * rand() / (double(RAND_MAX)))) / 180 * M_PI * 5 / 6;
				m4++;
				break;
			default:
				dis = min + (max - min) * (rand() / double(RAND_MAX));
				if (k % 3 != 0)
					deltaangle = (-180 + (360 * rand() / (double(RAND_MAX)))) / 180 * M_PI_2;
				else
					deltaangle = (-180 + (360 * rand() / (double(RAND_MAX)))) / 180 * M_PI;
				break;
			}

			currentangle = lastangle + deltaangle;
			if (currentangle < 0)
				currentangle = M_PI * 2 + currentangle;
			else if(currentangle > M_PI * 2 )
			{
				currentangle = currentangle - M_PI * 2;
			}

			nextdata.x = currentdata.x + dis * cos(currentangle);
			nextdata.y = currentdata.y + dis * sin(currentangle);
			
			/*
			if (currentangle > M_PI) {
				currentangle = currentangle - M_PI * 2;
			}
			else if (currentangle < -M_PI)
			{
				currentangle = currentangle + M_PI * 2;
			}

			nextdata.x = currentdata.x + dis * cos(TransferAngle(currentangle));
			nextdata.y = currentdata.y + dis * sin(TransferAngle(currentangle));

			*/

			//std::cout << lastangle << " " << currentangle << " " << deltaangle << std::endl;

			lastdata = currentdata;
			lastangle = currentangle;

			//std::cout << lastangle << " " << currentangle << " " << deltaangle << std::endl;

		}
	}

	std::cout << m1 << " " << m2 << " " << m3 << " " << m4 << std::endl;

}



bool GenerateTrainData(string inputFileName, int num, string outputFileName, vector<int>& TrainTypeNum, vector<int>& TestTypeNum, int train_id_num, int test_id_num, int pNum) {
	fstream f;
	ofstream of;
	ofstream tf;
	ofstream opf;
	ofstream tpf;


	//of.open(outputFileName + "/traindata.txt", ios::app);
	//tf.open(outputFileName + "/testdata.txt", ios::app);
	//int TypeNum[6] = { 0 };
	//int tmp = 0;

	for (int k = 0; k < num; k++)
	{
		std::string name;
		int tmp = k;
		if (tmp == 0)
		{
			name.push_back('0');
		}
		else {
			while (tmp > 0)
			{
				name.push_back('0' + tmp % 10);
				tmp /= 10;
			}
			for (int i = 0; i < name.length() / 2; i++)
			{
				char t = name[i];
				name[i] = name[name.length() - i - 1];
				name[name.length() - i - 1] = t;
			}
		}
		vector<double> FL;
		vector <double> FP;
		f.open(inputFileName + "/" + name + ".txt", ios::in);
		if (!f.is_open()) {
			std::cout << "File open error!" << std::endl;
			break;
		}

		int n = 0;
		while (n < pNum) {
			//	std::string tmp;
			//	std::getline(f, tmp);
			double x, y;
			f >> x >> y;
			FP.push_back(x);
			FP.push_back(y);
			n++;
		}

		double label;
		f >> label;
		f.close();

		int id = label - 1;
		if (TrainTypeNum[id] < train_id_num) {

			string name = NumToStr(id) + NumToStr(TrainTypeNum[id]);

			opf.open(outputFileName + "/traincrv" + "/" + name + ".txt", ios::app);

			opf << FP.size() / 2 << "\n";
			for (int j = 0; j < FP.size(); j += 2)
			{
				opf << FP[j] << " " << FP[j + 1] << "\n";
			}
			opf.close();
			TrainTypeNum[id]++;

		}
		else if (TestTypeNum[id] < test_id_num) {

			string name = NumToStr(id) + NumToStr(TestTypeNum[id]);

			tpf.open(outputFileName + "/testcrv" + "/" + name + ".txt", ios::app);

			tpf << FP.size() / 2 << "\n";
			for (int j = 0; j < FP.size(); j += 2)
			{
				tpf << FP[j] << " " << FP[j + 1] << "\n";
			}

			tpf.close();
			TestTypeNum[id]++;
		}
		else {
			bool tmp = true;
			for (int j = 3; j < TrainTypeNum.size(); j++)
			{
				if (TrainTypeNum[j] < train_id_num)
					tmp = false;
				if (TestTypeNum[j] < test_id_num)
					tmp = false;
			}
			if (tmp)
			{
				return true;
			}

		}
	}

	for (int i = 0; i < 6; i++)
	{
		std::cout << TrainTypeNum[i] << " ";

	}
	std::cout << "\n";
	for (int i = 0; i < 6; i++)
	{
		std::cout << TestTypeNum[i] << " ";

	}

	return false;
}

