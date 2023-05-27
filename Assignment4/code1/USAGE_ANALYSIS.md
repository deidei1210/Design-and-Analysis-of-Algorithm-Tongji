# code1_Usage

## 开发环境

- VScode
- Python 3.8.2
- 需要安装的包：opencv , numpy

## 代码目标

本代码基于动态规划实现了Seam Carving（缩减图像）算法，可以将指定的图片进行缩放。

与传统的缩放算法不同，Seam Carving 可以在保留图像主体结构的同时，针对性地削减图像中的细节部分。在缩小图像的过程中，Seam Carving 会删除图像中的某些像素行或列，从而实现对图像的缩小。

具体而言，Seam Carving 算法的核心思想是将图像看作一个**能量分布的矩阵**，每个像素都有一个对应的能量值。我们可以通过计算每个像素与其周围像素的差异或梯度等方式来确定每个像素的能量值。

在找到图像中能量值最小的像素路径（Seam）之后，我们可以将其删除，从而得到一个缩小了一行或一列的新图像。重复执行这一过程，直到达到所需的图像大小为止。

与简单的裁剪操作相比，Seam Carving 可以更加智能地调整图像大小，避免了因简单剪切而导致的失真和变形。Seam Carving 的应用非常广泛，比如可以用于手机屏幕、电脑屏幕和互联网上的图片等等场景中。

## 项目结构

```python
code1/
    ├── example/
    │   ├── image.jpg
    │   ├── image_result.jpg
    │   ├── image6.jpg
    │   ├── image6_result.jpg
    ├── SeamCarving.py
    ├── USAGE_ANALYSIS.md
```

- example  目录用于存储需要进行操作的图片。其中image.jpg与image6.jpg为缩放前的图片，image_result.jpg与image6_result.jpg为缩放后的图片，
- SeamCarving.py 是我实现SeamCarving的代码。
- USAGE_ANALYSIS.md 是项目运行说明以及算法时间、空间复杂度分析。

## 运行方法

- 将code1文件夹在VScode中打开。
- 将SeamCarving.py中 input_image 的值替换成想要缩放的图片的路径。然后将 output_image 的值替换成输出的路径。

```abap
input_image="example/image.jpg"            #需要进行缩放的图片路径（相对路径）
output_image="example/image_result.jpg"   #处理后图片的输出路径
```

- 设置缩放比例

```abap
#指定期望的resize之后图片的长和宽占原来的百分比
height_percent_resize=1  #保持高度不变
width_percent_resize=0.5 #宽度减少为原来的二分之一
```

- 右击VScode空白区域，在终端运行文件。

## 时间&空间复杂度分析

 Seam Carving的**时间复杂度**为 ***O*( *m n* )**，其中 m 和 n 分别代表输入矩阵 `energy_map` 的行数和列数。

空间复杂度主要来自两部分。第一部分是创建了一个副本 `cumulative_map` 的矩阵，其大小与输入矩阵 `energy_map` 相同，因此消耗的空间大小为 ***O*(*mn*)**。第二部分是创建了一个大小为 `(m,)` 的一维数组 `output`，用于存储像素点所在的缝隙路径，因此空间复杂度为 *O*(*m*)。综上所述，该代码的**空间复杂度也为*O*(*mn*)**。

需要注意的是，虽然该算法的时间复杂度和空间复杂度都为 *O*(*mn*)，但实际运行时会受到常数因子等因素的影响。特别是对于大规模的输入矩阵，可能会导致计算时间非常长，甚至无法完成。因此，在实际应用中，我们需要针对具体问题进行调优，以提高算法的效率。

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

## 运行效果

![运行结果图1](https://github.com/deidei1210/Design-and-Analysis-of-Algorithm-Tongji/blob/master/Assignment4/SC/code1/%E8%BF%90%E8%A1%8C%E6%95%88%E6%9E%9C%E5%9B%BE/Untitled%201.png)

（左图为缩放前的图像，右图为宽缩放了二分之一的图像）

![运行结果图2](https://github.com/deidei1210/Design-and-Analysis-of-Algorithm-Tongji/blob/master/Assignment4/SC/code1/%E8%BF%90%E8%A1%8C%E6%95%88%E6%9E%9C%E5%9B%BE/Untitled.png)