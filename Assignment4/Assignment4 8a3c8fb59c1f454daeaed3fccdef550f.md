# Assignment4


---

1. （**10 分）最短路径数量** 国际象棋中的车可以水平或竖直移到棋盘中同行或同列的任何

一格。将车从棋盘的一角移到另一对角，有多少条最短路径？路径的长度由车所经过的

方格数（包括第一格和最后一格）来度量。使用下列方法求解该问题。

a) 动态规划算法

b) 基本排列组合

**a) 动态规划算法** 

假设棋盘的大小为n*n，车在初始状况下位于棋盘的左上角，如果要将车从棋盘的一角移动到另外一角，并且**路径最短，则车在每一步要么向下移动，要么向右移动**，不能往回走。

我们假设**F[i , j]** 表示车从坐（0,0）到（i,j）的方法数。

则由于当车走到（i,j）的时候，他的上一步所在格子要么是（i-1,j），要么是（i,j-1）。

所以 我们就将问题描述为了交叠子问题：

**F[ i , j ]=F[ i-1 , j ]+F[ i , j-1 ]**

该问题的初始边界条件为：

**F[0 , j] = 1 ，F[i , 0] = 1**

由最小边界条件依次递归，我们可以得到表格如下所示：

| 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | …… |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | …… |
| 1 | 3 | 6 | 10 | 15 | 21 | 28 | 36 | 45 | …… |
| 1 | 4 | 10 | 20 | 35 | 56 | 84 | 120 | 165 | …… |
| 1 | 5 | 15 | 35 | 70 | 126 | 210 | 330 | 495 | …… |
| 1 | 6 | 21 | 56 | 126 | 252 | 462 | 792 | 1287 | …… |
| 1 | 7 | 28 | 84 | 210 | 462 | 924 | 1716 | 3019 | …… |
| 1 | 8 | 36 | 120 | 330 | 792 | 1716 | 3432 | 6451 | …… |
| 1 | 9 | 45 | 165 | 495 | 1287 | 3003 | 6435 | 12886 | …… |
| 1 | 10 | 55 | 220 | 715 | 2002 | 5005 | 6935 | 19821 | …… |
| …… | …… | …… | …… | …… | …… | …… | …… | …… | …… |

伪代码如下所示：

```cpp
CalculateMinPathNum(n)
  //初始边界条件**F[0 , j] = 1 ，F[i , 0] = 1**
	for i<-0 to n-1 do
		F[i][0]<-0
    for j<-0 to n-1 do
	    F[0][j]<-0

	//开始进行递归
    for i<-1 to n-1 do
	    for j<-1 to n-1 do
		    F[i,j]=F[i-1,j]+F[i,j-1]
    //返回最后的结果
    return F[n-1][n-1]
```

事实上，由于题目中说是国际象棋，而国际象棋的棋盘尺寸为8✖️8，所以n=8。

**当n=8时，最短路径的个数为3432条路径。即本题的答案。**

**b）基本排列组合**

该问题的最短路径对应的移动次数是14次移动。无论是哪一条路径都需要经过**7次向下（down）** 👇的移动，以及**7次向右(right)** 👉的移动。但是向下和向右移动的先后顺序是任意的。即可以是（down，down，down，down，down，down，down，right，right，right，right，right，right，right，），也可以是（right，down，down，right，down，down，down，right，right，down，right，right，down，right，）。

因此，不同的最短路径数等同于从14次移动中选择7次向下（down）移动的次数。所以为C(14,7)次。计算得：

$$
C_{14}^{7}=3432
$$

所以，最短路径的个数是3432条。

---

![Untitled](Assignment4%208a3c8fb59c1f454daeaed3fccdef550f/Untitled.png)

 **a.** 对于此时的概率我们将其按照从小到大的顺序进行排序：

**B(0.1) < D(0.15) < - (0.15) < C(0.2) < A(0.4)**

然后我们先取其中概率最低的两个放入霍夫曼树：

![Untitled](Assignment4%208a3c8fb59c1f454daeaed3fccdef550f/Untitled%201.png)

于是生成一个新节点0.25，放入原来的字符概率中继续排序：

**- (0.15) < C(0.2) < 0.25 < A(0.4)**  

![Untitled](Assignment4%208a3c8fb59c1f454daeaed3fccdef550f/Untitled%202.png)

于是生成一个新节点0.35，放入原来的字符概率中继续排序：

**0.25 < 0.35 < A(0.4)**  

![Untitled](Assignment4%208a3c8fb59c1f454daeaed3fccdef550f/Untitled%203.png)

所以我们得到霍夫曼树为：

![Untitled](Assignment4%208a3c8fb59c1f454daeaed3fccdef550f/Untitled%204.png)

因此，我们可以给字符编码如下：

| 字符 | A | B | C | D | - |
| --- | --- | --- | --- | --- | --- |
| 概率 | 0.4 | 0.1 | 0.2 | 0.15 | 0.15 |
| 编码 | 0 | 100 | 111 | 101 | 110 |

b. 对文本ABACABAD进行编码：

**0100011101000101**

c. 利用我们在a小题中创建的霍夫曼编码，我们可以对下面内容进行解码：

**100 ｜ 0 ｜101 ｜ 110 ｜ 0 ｜ 101 ｜ 0**

**B       A       D            _         A       D       A**

所以为**BAD_ADA**

---

## 编程题思路

### 1. Seam Carving算法（python）

### 1.1Seam Carving简介

Seam Carving（缩减图像）是一种用于改变图像大小的算法。与传统的缩放算法不同，Seam Carving 可以在保留图像主体结构的同时，针对性地削减图像中的细节部分。在缩小图像的过程中，Seam Carving 会删除图像中的某些像素行或列，从而实现对图像的缩小。

具体而言，Seam Carving 算法的核心思想是将图像看作一个**能量分布的矩阵**，每个像素都有一个对应的能量值。我们可以通过计算每个像素与其周围像素的差异或梯度等方式来确定每个像素的能量值。

在找到图像中能量值最小的像素路径（Seam）之后，我们可以将其删除，从而得到一个缩小了一行或一列的新图像。重复执行这一过程，直到达到所需的图像大小为止。

与简单的裁剪操作相比，Seam Carving 可以更加智能地调整图像大小，避免了因简单剪切而导致的失真和变形。Seam Carving 的应用非常广泛，比如可以用于手机屏幕、电脑屏幕和互联网上的图片等等场景中。

### 1.2 Seam Carving实现

在本次作业中，我们的目标是读入目标图片，然后将目标图片的宽缩小到原来的二分之一大小。

**在我的代码中，使用者可以通过修改  input image 的值来读入需要操作的图片。同时，也可以把 output_image 的值修改成处理后图片输出的地址。**

**同时，可以调整height_percent_resize和width_percent_resize的值，来设置缩放的比例。**（比如说这边我们希望把原图的高不变，宽变成原来的0.5倍就可以设置成如下）

```python
#指定期望的resize之后图片的长和宽占原来的百分比
height_percent_resize=1  #保持高度不变
width_percent_resize=0.5 #宽度减少为原来的二分之一

input_image="example/image.jpg"            #需要进行缩放的图片路径（相对路径）
output_image="example/image_result.jpg"   #处理后图片的输出路径

#读入图片，并且将其按照64位浮点数存储
read_image = cv2.imread(input_image).astype(np.float64)
in_height, in_width, channels = read_image.shape # 获取图像的高度和宽度
out_height = height_percent_resize * in_height #计算期望输出的图片的高度
out_width = width_percent_resize * in_width #期望输出的图片的宽度
```

- 经过了初始化之后，我们进行Seam Carving的操作。

```python
#用来实现seams_carving的算法部分
def seamCarving():
    global out_image
    global col_change,col_add
    global row_change,row_add

    # 如果输出的宽度小于输入的宽度，说明图片需要减少列数（所以要移除像素）
    if col_change!=0:
        if not col_add:
            RemoveSeams(col_change * -1)
        # 如果输出的宽度大于输入的宽度，说明图片需要增加列数（所以要增加像素）
        elif col_add:
            InsertSeams(col_change)

    #处理完了列方向上的变化之后再处理行方向上的变化，处理行方向的变化可以先把图片旋转90度
    if row_change!=0:  #如果行方向上有变化就旋转，否则不用处理这个
        out_image = RotateImage(out_image, 1)
        # 如果输出的高度小于输入的高度，说明图片需要减少行数（所以要移除像素）
        if not row_add:
            RemoveSeams(row_change * -1)
        # 如果输出的高度大于输入的高度，说明图片需要增加行数（所以要增加像素）
        elif row_add:
            InsertSeams(row_change)
        #处理完之后再转回来
        out_image = RotateImage(out_image, 0)
```

在这段进入Seam Carving算法的代码之中，我们首先先对其宽度进行处理，然后对其高度进行处理。

**如果col_change不等于0，则说明图片在宽度上（即像素矩阵的列）需要有改变。** 如果是增加列数，那么执行InsertSeams函数；否则，说明需要减少列数，就应该执行RemoveSeams函数。
1. **如果row_change不等于0，则说明图片在高度上（即像素矩阵的行）需要有改变。** 但是，为了和上面对列的操作函数统一（少写几个函数），我们先将图片旋转90度，这样行操作就变成了列操作。接下来类似的，如果需要增加行，那就InsertSeams，否则RemoveSeams。

- 那么，如果是将宽度缩小为二分之一，我们如何判断应该删除哪些像素呢？我们肯定是需要删除那些不那么重要的像素，这样才能避免图像的失真。我们首先计算能量矩阵（energy_map）来衡量每一个像素的重要程度。

```python
#计算能量矩阵，能量矩阵中使用了Scharr算子来检测图像的边缘，
#能量矩阵中能量越高的像素点表示其越重要，不能够被删除
def CalculateEnergyMap():
    global out_image
    #由于cv2读入图片之后存储颜色时通道为bgr，而不是rgb，所以要格外小心顺序不能反
    b, g, r = cv2.split(out_image)
    b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) \
               + np.absolute(cv2.Scharr(b, -1, 0, 1))
    g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0))\
							 + np.absolute(cv2.Scharr(g, -1, 0, 1))
    r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) \
								+ np.absolute(cv2.Scharr(r, -1, 0, 1))
    return b_energy + g_energy + r_energy
```

这段代码使用了 Scharr 算子来计算图像的边缘，并根据得到的边缘信息来计算能量矩阵。

它将读入的彩色图像 `out_image` 拆分成三个通道，即红色通道、绿色通道和蓝色通道。然后对每个通道分别使用 Scharr 算子进行卷积操作，在水平和垂直方向上检测图像的边缘。

在对每个通道完成边缘检测之后，可以得到三个能量矩阵，分别代表了在每个通道中像素点的重要程度。这里采用了将能量矩阵的值相加的方式，得到总的能量矩阵。在能量矩阵中，像素点的能量越高，表示其越重要，不能够被删除，从而为 Seam Carving 算法提供重要的能量信息。

- 在计算出**能量矩阵（energy_map）** 之后，便需要在能量矩阵中找到能量之和最小的路径，如果删除这一条路径，那么对于图片整体的影响就是最小的。

我们可以利用动态规划的方法来寻找这个最短路径。

我们假设一个二维数组 **M ( i , j ),** 其中 i 和 j 分别表示图片像素的行和列。M ( i , j )中存储的数字表示从图片的顶上到当前位置可能的能量累积的最小值。我们知道从上往下走，从当前格子往上看，能够连通的格子只可能为**左上M(i-1,j-1)，上面M(i-1,j)，以及右上M(i-1,j+1)的格子。** 因此，当前格子的能量路径最小值应该为状态转移方程如下所示：

**M( i , j )=e( i , j )+min⁡( M( i-1 , j-1 ),M( i-1 , j ),M( i-1 , j+1 ) )**

其中可以直接初始化顶上第一行的M ( i , j )为energy_map对应的第一行的值。另外，如果M ( i , j )对应像素如果位于图像最左边，则不用考虑**M( i-1 , j-1 ) ，**如果对应像素位于图像的最右边，则不用考虑**M( i-1 , j+1 ) 。**

```python
# 动归：M(i,j)=e(i,j)+min⁡(M(i-1,j-1),M(i-1,j),M(i-1,j+1))
# 动归每一行每一列，计算每行(i, j)所有可能连通接缝的累计最小能量M
def CumulativeMapForward(energy_map):   
    m, n = energy_map.shape
    #我们使用 np.copy() 函数来避免直接修改 energy_map 对象本身，因为 energy_map 可能会被其他地方的代码所使用，如果直接修改 energy_map，可能会对其他部分的程序产生影响。所
    # 以，通过创建一个副本 cumulative_map 来进行操作，可以避免这种情况的发生。
    cumulative_map = np.copy(energy_map)

    # 动归：M(i,j)=e(i,j)+min⁡(M(i-1,j-1),M(i-1,j),M(i-1,j+1))
    #递归每一行每一列，计算每行(i, j)所有可能连通接缝的累计最小能量M
    for row in range(1, m):     
        for col in range(n):
            #如果是最左边一列,那么当前像素能量累积路径最小的大小应该为其右边和上面中较小的值加上其当前的值
            if col == 0:
                cumulative_map[row, col] = energy_map[row, col] + min(energy_map[row-1, col+1], energy_map[row-1, col])
            #如果是最右边的一列
            elif col == n - 1:
                cumulative_map[row, col] = energy_map[row, col] + min(energy_map[row-1, col-1], energy_map[row-1, col])
            #如果是中间的部分
            else:
                cumulative_map[row, col] = energy_map[row, col] + min(energy_map[row-1, col+1], energy_map[row-1, col], energy_map[row-1, col-1])
    
    m, n = cumulative_map.shape
    output = np.zeros((m,), dtype=np.uint32)
    output[-1] = np.argmin(cumulative_map[-1]) #在最后一行（即图像底部）寻找累积能量值最小的像素点，并保存该点的列索引值为 output[-1]。

    # 从倒数第二行开始，依次向上扫描每一行。对于每一行，都根据下一行中与上一个像素点和当前像素点相邻的三个像素的累积能量值，
    # 选择其中最小的一个像素点作为当前像素点所在的缝隙位置，并将该点的列索引值保存到结果数组 output 中。
    # 这段代码的作用是根据下一行已经选取的像素点的列索引值，在当前行中选取一个符合条件的像素点，
    # 并将其列索引值保存到结果数组 output 中，用于最终生成整幅图像的缝隙路径。
    for row in range(m - 2, -1, -1):
        seam_col = output[row + 1]
        if seam_col == 0:
            output[row] = np.argmin(cumulative_map[row, : 2])
        else:
            output[row] = np.argmin(cumulative_map[row, seam_col - 1: min(seam_col + 2, n - 1)]) + seam_col - 1
    return output
```

在计算好了cumulative_map之后，我们可以通过回溯法，从下往上寻找出每一行在能量最小的路径上的像素点，并将其col列数记录在output数组中。（在output数组中，下标表示当前的行数，数组的内容表示在当前行并且在能量最小路径上的像素的col值）

于是，在计算出了当前能量最小的路径之后，我们就可以删除这条Seam，然后继续新一轮的计算：

CalculateEnergyMap()→CumulativeMapForward(energy_map)→DeleteSeam(seam_idx)

### 1.3 Seam Carving 时间复杂度&空间复杂度

 Seam Carving的**时间复杂度**为 ***O*( *m n* )**，其中 m 和 n 分别代表输入矩阵 `energy_map` 的行数和列数。

空间复杂度主要来自两部分。第一部分是创建了一个副本 `cumulative_map` 的矩阵，其大小与输入矩阵 `energy_map` 相同，因此消耗的空间大小为 ***O*(*mn*)**。第二部分是创建了一个大小为 `(m,)` 的一维数组 `output`，用于存储像素点所在的缝隙路径，因此空间复杂度为 *O*(*m*)。综上所述，该代码的**空间复杂度也为*O*(*mn*)**。

需要注意的是，虽然该算法的时间复杂度和空间复杂度都为 *O*(*mn*)，但实际运行时会受到常数因子等因素的影响。特别是对于大规模的输入矩阵，可能会导致计算时间非常长，甚至无法完成。因此，在实际应用中，我们需要针对具体问题进行调优，以提高算法的效率。

### 1.4 运行效果

![Untitled](Assignment4%208a3c8fb59c1f454daeaed3fccdef550f/Untitled%205.png)

（左图为缩放前的图像，右图为宽缩放了二分之一的图像）

![Untitled](Assignment4%208a3c8fb59c1f454daeaed3fccdef550f/Untitled%206.png)

---

## 2.  梯度下降法求函数最小值

### 2.1 问题描述

在求解复杂最优化问题，尤其是非凸函数时，常常无法通过直接计算导数为0 来计算出最值点，此时，经常会用到梯度下降法来代替。

**梯度下降法的思想：** 通过将一步计算分解成多步计算，不断迭代进行近似求解。

**梯度下降法的迭代公式：** 𝒙𝑘+1 = 𝒙𝑘 − 𝑡∇𝑓(𝒙)

（其中∇𝑓(𝒙)是当前点的梯度，𝑡是步长(学习率)，迭代预先设定的迭代次数 n 后，就得到了算法的解）

现在通过python来编程，基于梯度下降法来求解下列函数的最小值，以及相应的(x1,x2)。

$$f(x_1,x_2) = e^{x_1+3x_2-0.1} + e^{x_1-3x_2-0.1} + e^{-x_1-0.1}$$

### 2.2 代码实现

首先定义一个函数来实现函数求值：

```python
# 定义函数 f(x)
def f(x):
    return np.exp(x[0]+3*x[1]-0.1) + np.exp(x[0]-3*x[1]-0.1) + np.exp(-x[0]-0.1)
```

然后我们可以计算出，该函数的梯度为：

$$\nabla f(x) = \begin{bmatrix}e^{x_1+3x_2-0.1} + e^{x_1-3x_2-0.1} - e^{-x_1-0.1} \\3e^{x_1+3x_2-0.1} - 3e^{x_1-3x_2-0.1}\end{bmatrix}$$

所以定义一个函数来计算其梯度：

```python
# 定义函数 f(x) 的梯度 grad_f(x)
def grad_f(x):
    return np.array([
        np.exp(x[0]+3*x[1]-0.1) + np.exp(x[0]-3*x[1]-0.1) - np.exp(-x[0]-0.1),
        3*np.exp(x[0]+3*x[1]-0.1) - 3*np.exp(x[0]-3*x[1]-0.1)
    ])
```

最后定义梯度下降函数：

其中我们设定函数的参数：

- f：要最小化的目标函数
- grad_f：目标函数的梯度
- x0：优化变量的初始值
- learning_rate：学习率，控制每次更新的步长（默认为0.01）
- epochs：最大迭代次数（默认为100000）
- eps：当两个相邻的迭代点之间的距离小于 eps 时停止迭代（默认为1e-10）

```python
def gradient_descent(f, grad_f, x0, learning_rate=0.01, epochs=100000, eps=1e-10):
    x = x0
    for k in range(epochs):
        # 计算当前点的梯度
        grad = grad_f(x)
        # 计算下一步的迭代点
        x_new = x - learning_rate * grad
        # 计算两个相邻迭代点之间的距离差
        diff = np.linalg.norm(x_new - x)
        # 如果距离差小于 eps，停止迭代
        if diff < eps:
            # print(diff)
            # print(eps)
            break
        else:
            # 否则继续迭代
            x = x_new
    # 返回最终迭代到的位置和对应的函数值（即极小值）
    return x, f(x)
```

### 2.3 运行结果

![Untitled](Assignment4%208a3c8fb59c1f454daeaed3fccdef550f/Untitled%207.png)
