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
	//m��������n������
	int m = matrix.size(), n = matrix[0].size();
	//�ӵ�һ�����һ�п�ʼ
	int i = 0, j = n - 1;
	while (i < m && j >= 0){
		//����ҵ��˾ͷ���true
		if (matrix[i][j] == target){
			return true;
		}
		//�����Ŀ������������Ѱ��
		else if (matrix[i][j] > target){
			j--;
		}
		//�����Ŀ����С��������Ѱ��
		else{
			i++;
		}
	}
	//û�ҵ�
	return false;
}

int main()
{

	//	��ȡ�������� 
	ifstream inFile("testcase.csv", ios::in);
	string lineStr;
    //��ʼ��ʱ
	start_ = clock();
    //�ж��ļ��Ƿ�ɹ���
	if (!inFile.is_open())
	{
		cout << "Error!" << endl;
	}

	//	���Խ�����
	int correct_num = 0;
	int error_num = 0;

	//	���в������ݣ�������
	while (getline(inFile, lineStr))  //һ��һ�ж�
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
	cout << "��ʱ:" << endtime * 1000 << "ms" << endl;

	system("pause");

	return 0;
}
