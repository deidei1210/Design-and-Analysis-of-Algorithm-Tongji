#include<iostream>
using namespace std;
//一次可以爬1级台阶，或者2级台阶
//爬1级台阶：消耗1个能量，爬2级台阶：消耗3个能量
//输入：m级台阶,n个能量
//输出：有多少种爬台阶的方法

int solution(int stairs,int calories){
    //如果输入的能量小与台阶的数量，或者输入非法值，则返回0种方案
    if(calories<stairs||calories<=0||stairs<=0)
        return 0;
    else if(calories==stairs)//如果楼梯数和台阶数相同，则只有一种方案
        return 1;

    int dp[1000][1000]={};//动态规划的数组，前一个下标表示台阶的数量，后面一个下标表示剩下能量的多少
    //给出初始条件
    dp[0][0]=0;      
    //只剩下一级台阶的时候一定只有一种方法
    for(int j=1;j<1000;j++){  
        dp[1][j]=1;
    }
    //如果还剩下两级台阶，且能量剩下的大于等于3，则一定有两种方式
    for(int j=3;j<1000;j++){ 
        dp[2][j]=2;
    }
    dp[2][2]=1;
    
    for(int i=3;i<=stairs;i++){
        for(int j=i;j<=calories;j++){
            dp[i][j]=dp[i-1][j-1]+dp[i-2][j-3];
        }
    }
    return dp[stairs][calories];
}

int main()
{
    int m=0,n=0;
    cin>>m>>n;
    cout<<solution(m,n)<<endl;
    return 0;
}

