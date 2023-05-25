#include<iostream>
using namespace std;

//计算阶乘
int factorial(int n){
    int result=1;
    for(int i=n;i>=1;i--){
        result*=i;
    }
    return result;
}


int solution(int stairs,int cal){
    //如果楼梯数量大于能量，或者其中有一个小于等于0，则返回0
    if(stairs>cal||stairs<=0||cal<=0)
        return 0;
    else if(stairs==cal)//若楼梯数量和能量大小相等，则只有一种情况
        return 1;
    
    int stairs_half=stairs/2;
    int two=0,one=0;   //two表示一次性上两节台阶的次数，one表示一次性上一个台阶的次数
    for(int i=stairs_half;i>=0;i--){
        //如果这个次数的一次性上两节台阶可以被cal支持，就结束循环
        if(3*i+stairs-2*i<=cal){
            two=i;
            one=stairs-i*2;
            break;
        }
    }
    //于是我们得到一次性上两节台阶的次数是two，一次性上一节台阶的次数是one
    //接下来进行排列组合，求出在这种情况下一共有多少种不同的组合方式
    int item=two+one;
    int sum=factorial(item)/(factorial(two)*factorial(one));

    return sum;
}
int main()
{
    int m=0,n=0;
    cin>>m>>n;
    cout<<solution(m,n)<<endl;
    return 0;
}

