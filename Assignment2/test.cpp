// -*- coding: utf-8 -*-
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <ctime>

clock_t start_, end_;

using namespace std;

bool searchMatrix(vector< vector<int> > &matrix, int target)
{
	// TODO
	if (matrix.empty() || matrix[0].empty()){
		return false;
	}
	//m是行数，n是列数
	int m = matrix.size(), n = matrix[0].size();
	//从第一行最后一列开始
	int i = 0, j = n - 1;
	while (i < m && j >= 0){
		//如果找到了就返回true
		if (matrix[i][j] == target){
			return true;
		}
		//如果比目标数大，则向左寻找
		else if (matrix[i][j] > target){
			j--;
		}
		//如果比目标数小，则向下寻找
		else{
			i++;
		}
	}
	//没找到
	return false;
}

int main()
{

	//	读取测试数据 
	ifstream inFile("testcase.csv", ios::in);
	string lineStr;
    //开始计时
	start_ = clock();
    //判断文件是否成功打开
	if (!inFile.is_open())
	{
		cout << "Error!" << endl;
	}

	//	测试结果标记
	int correct_num = 0;
	int error_num = 0;

	//	运行测试数据，输出结果
	while (getline(inFile, lineStr))  //一行一行读
	{
		vector< vector<int> > matrix;

		string number;
		bool num_end = false;
		bool line_end = false;
		int target = -1;
		bool result;
		vector<int> line;
		for (int i = 0; i < lineStr.size(); i++){
			if (!num_end){
				if (lineStr[i] == '['){
					line_end = false;
					line.clear();
				}
				else if (lineStr[i] == ']' && line_end){
					number = "";
					num_end = true;
				}
				else if (lineStr[i] == ']' && !num_end){
					line.push_back(atoi(number.c_str()));
					matrix.push_back(line);
					line_end = true;
					number = "";
				}

				else if (lineStr[i] >= '0' && lineStr[i] <= '9')
					number += lineStr[i];
				else if (lineStr[i] == ',' && !line_end){
					line.push_back(atoi(number.c_str()));
					number = "";
				}
			}
			else{
				if (target == -1){

					if (lineStr[i] >= '0' && lineStr[i] <= '9')
						number += lineStr[i];
					else if (lineStr[i] == ',' && number != "")
						target = atoi(number.c_str());
				}
				else{
					result = lineStr[i] - '0';
				    // cout<<int(lineStr[i])<<endl;
				}
			}
		}
		// cout<<searchMatrix(matrix, target)<< " "<<result<<endl;
		// cout<<target<<endl;
		if (result == searchMatrix(matrix, target))
			correct_num += 1;
		else
			error_num += 1;
	}
	end_ = clock();
	double endtime = (double)(end_ - start_) / CLOCKS_PER_SEC;
	inFile.close();

	cout << "correct:" << correct_num << endl;
	cout << "error:" << error_num << endl;
	cout << "用时:" << endtime * 1000 << "ms" << endl;

	system("pause");

	return 0;
}
