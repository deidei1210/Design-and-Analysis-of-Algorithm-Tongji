#include<iostream>
#include<vector>

using namespace std;

//枚举出矿工行走的四个方向
const int dirs[4][2] = { {-1,0},{1,0},{0,-1},{0,1} }; //分别为下，上，左，右
vector<vector<int>> GoldMap;    //一个二维数组，用来表示每个格子里面金子的数量
int max_value = 0;
int row = 0, col = 0;    //金矿的行列

//x,y表示当前矿工所在的位置，矿工从当前位置开始开采
//gold为矿工在对x,y处开采之前已经拥有的黄金数量
void dfs(int x, int y, int gold) {
    gold += GoldMap[x][y];    //矿工对x,y处进行开采，已有的黄金数量加上x,y处的黄金数量
    max_value = max(max_value, gold);  //将黄金的最大数量进行更新

    int record = GoldMap[x][y];
    GoldMap[x][y] = 0;

    //矿工在开采完了这一个格子内的黄金之后开始向下一个格子行走，枚举四个方向
    for (int d = 0; d < 4; ++d) {
          int nx = x + dirs[d][0];
          int ny = y + dirs[d][1];
          //如果没有走出金矿的范围，并且该格子内有金矿，那就继续dfs，因为可以继续采矿
          if (nx >= 0 && nx < row &&
              ny >= 0 && ny < col 
              && GoldMap[nx][ny] > 0) {
                   dfs(nx, ny, gold);
          }
    }
    //将该格子恢复原状
    GoldMap[x][y] = record;
}

int solution(vector<vector<int>> GoldMap) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            if (GoldMap[i][j] != 0) {
                dfs(i, j, 0);
            }
        }
    }
    return max_value;
}
int main()
{
    int gold_num = 0;        //每个格子中金矿的数量
    vector<int> GoldMap_row;//每一个行的金块数量数组

    cout << "请输入金矿的行数：";
    cin >> row;
    cout << "请输入金矿的列数：";
    cin >> col;

    for (int i = 0; i < row; i++) {
        cout << "请输入第" << i + 1 << "行的内容：";
        for (int j = 0; j < col; j++) {
            cin >> gold_num;
            GoldMap_row.push_back(gold_num);
        }
        GoldMap.push_back(GoldMap_row);
        GoldMap_row.clear();
    }
    cout << solution(GoldMap) << endl;

    return 0;
}