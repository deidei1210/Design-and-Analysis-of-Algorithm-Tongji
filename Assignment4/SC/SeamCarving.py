import numpy as np
import cv2

#指定缩放后图片的高度和宽度
new_height = 312
new_width = 312

#指定期望的resize之后图片的长和宽占原来的百分比
height_percent_resize=1  #保持高度不变
width_percent_resize=0.5 #宽度减少为原来的二分之一

input_image="example/image.jpg"            #需要进行缩放的图片路径（相对路径）
output_image="example/image_result.jpg"   #处理后图片的输出路径

#输出图片路径，检查是否正确
print(input_image)
print(output_image)

#读入图片，并且将其按照64位浮点数存储
read_image = cv2.imread(input_image).astype(np.float64)

# 获取图像的高度和宽度
in_height, in_width, channels = read_image.shape

print(in_height,in_width)

#计算期望输出的图片的高度
out_height = height_percent_resize * in_height 
#期望输出的图片的宽度
out_width = width_percent_resize * in_width 

print(out_height,out_width)

# 计算需要被移走的像素的行数与列数
row_change, col_change = int(out_height - in_height), int(out_width - in_width)
#根据计算出来的行数的变化以及列数的变化来判断每个方向上是添加像素还是减少像素
row_add = row_change > 0
col_add=col_change>0

print(row_add,col_add)

#全局变量，在下面使用时需要申明
out_image = np.copy(read_image)

    #添加缝隙
def AddSeam(seam_idx):
    global out_image
    m, n = out_image.shape[: 2]
    output = np.zeros((m, n + 1, 3))
    for row in range(m):
        col = seam_idx[row]
        for ch in range(3):
            if col == 0:
                p = np.average(out_image[row, col: col + 2, ch])
                output[row, col, ch] = out_image[row, col, ch]
                output[row, col + 1, ch] = p
                output[row, col + 1:, ch] = out_image[row, col:, ch]
            else:
                p = np.average(out_image[row, col - 1: col + 1, ch])
                output[row, : col, ch] = out_image[row, : col, ch]
                output[row, col, ch] = p
                output[row, col + 1:, ch] = out_image[row, col:, ch]
    out_image = np.copy(output)

def UpdateSeams(remaining_seams, current_seam):
    output = []
    for seam in remaining_seams:
        seam[np.where(seam >= current_seam)] += 2
        output.append(seam)
    return output

#删除缝隙
def DeleteSeam(seam_idx):
    # global out_image = np.copy(in_image)
    # print("Start delete seams!")
    global out_image
    m, n = out_image.shape[: 2]
    output = np.zeros((m, n - 1, 3))
    for row in range(m):
        col = seam_idx[row]
        output[row, :, 0] = np.delete(out_image[row, :, 0], [col])
        output[row, :, 1] = np.delete(out_image[row, :, 1], [col])
        output[row, :, 2] = np.delete(out_image[row, :, 2], [col])
    out_image = np.copy(output)

#寻找符合条件的缝隙
def FindSeam(cumulative_map):
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
    return cumulative_map

def CumulativeMapBackward(energy_map):
    m, n = energy_map.shape
    output = np.copy(energy_map)
    for row in range(1, m):
        for col in range(n):
            output[row, col] = \
                energy_map[row, col] + np.amin(output[row - 1, max(col - 1, 0): min(col + 2, n - 1)])
    return output

#计算能量矩阵，能量矩阵中使用了Scharr算子来检测图像的边缘，能量矩阵中能量越高的像素点表示其越重要，不能够被删除
def CalculateEnergyMap():
    global out_image
    #由于cv2读入图片之后存储颜色时通道为bgr，而不是rgb，所以要格外小心顺序不能反
    b, g, r = cv2.split(out_image)
    b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
    g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
    r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
    return b_energy + g_energy + r_energy

#用来当宽度缩小或者高度缩小的时候移除缝隙
def RemoveSeams(total_times):
    print("Start RemoveSeams!")
    for i in range(total_times):
        energy_map = CalculateEnergyMap()   #计算能量矩阵
        cumulative_map = CumulativeMapForward(energy_map) 
        seam_idx = FindSeam(cumulative_map)  #找到每一行在Seam上的像素的列数
        DeleteSeam(seam_idx) #删除这些列

# 在需要增加宽度的时候增加缝隙的个数
def InsertSeams(total_times):
    global out_image
    temp_image = np.copy(out_image)
    seams_record = []
    print("Start InsertSeams!")
    for i in range(total_times):
        energy_map = CalculateEnergyMap()
        cumulative_map = CumulativeMapBackward(energy_map)
        seam_idx = FindSeam(cumulative_map)
        seams_record.append(seam_idx)
        DeleteSeam(seam_idx)

    out_image = np.copy(temp_image)
    n = len(seams_record)
    for i in range(n):
        seam = seams_record.pop(0)
        AddSeam(seam)
        seams_record = UpdateSeams(seams_record, seam)

# 将图片旋转90度
def RotateImage(image, choose_rotate_dir):
    #如果开始旋转图片输出提示
    print("Start rotate image!")
    m, n, channels = image.shape
    output = np.zeros((n, m, channels))
    if choose_rotate_dir:
        image_flip = np.fliplr(image)
        for c in range(channels):
            for row in range(m):
                output[:, row, c] = image_flip[row, :, c]
    else:
        for c in range(channels):
            for row in range(m):
                output[:, m - 1 - row, c] = image[row, :, c]
    return output

#用来实现seams_carving的算法部分
def seamCarving():
    global out_image
    global col_change,col_add
    global row_change,row_add
    #如果开始执行算法了，那就输出一些提示，并且检查col_change和row_change的大小是否正确
    print("Start Seam Carving!")
    print(col_change,row_change)
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

#保存图像的最终结果到指定地址
def SaveResult(image_path):
    global out_image
    cv2.imwrite(image_path, out_image.astype(np.uint8))

seamCarving()
SaveResult(output_image)