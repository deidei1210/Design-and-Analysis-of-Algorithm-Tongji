import numpy as np
import cv2


class SeamCarver:
    def __init__(self, image_path, height_percent_resize, width_percent_resize):
        # 初始化参数
        self.image_path = image_path     #需要处理的图片的路径
        
        # 读入图片，并且将图片以64位浮点数存储
        self.in_image = cv2.imread(image_path).astype(np.float64)
        self.in_height, self.in_width = self.in_image.shape[: 2]  #获得读入图片的高度和宽度
        self.out_height = height_percent_resize * self.in_height #期望输出的图片的高度
        self.out_width = width_percent_resize * self.in_width   #期望输出的图片的宽度
       
        self.out_image = np.copy(self.in_image)

        self.kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
        self.kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
        self.kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)

        # starting program
        self.seams_carving()

    #用来实现seams_carving的算法部分
    def seams_carving(self):
        # 计算需要被移走的像素的行数与列数
        delta_row, delta_col = int(self.out_height - self.in_height), int(self.out_width - self.in_width)

        # 如果输出的宽度小于输入的宽度，说明图片需要减少列数（所以要移除像素）
        if delta_col < 0:
            self.seams_removal(delta_col * -1)
        # 如果输出的宽度大于输入的宽度，说明图片需要增加列数（所以要增加像素）
        elif delta_col > 0:
            self.seams_insertion(delta_col)

        # 如果输出的高度小于输入的高度，说明图片需要减少行数（所以要移除像素）
        if delta_row < 0:
            self.out_image = self.rotate_image(self.out_image, 1)
            self.seams_removal(delta_row * -1)
            self.out_image = self.rotate_image(self.out_image, 0)
        # 如果输出的高度大于输入的高度，说明图片需要增加行数（所以要增加像素）
        elif delta_row > 0:
            self.out_image = self.rotate_image(self.out_image, 1)
            self.seams_insertion(delta_row)
            self.out_image = self.rotate_image(self.out_image, 0)

    #移除缝隙
    def seams_removal(self, num_pixel):
        for dummy in range(num_pixel):
            energy_map = self.calc_energy_map()                        #计算能量矩阵
            cumulative_map = self.cumulative_map_forward(energy_map)
            seam_idx = self.find_seam(cumulative_map)
            self.delete_seam(seam_idx)


    def seams_insertion(self, num_pixel):
        temp_image = np.copy(self.out_image)
        seams_record = []

        for dummy in range(num_pixel):
            energy_map = self.calc_energy_map()
            cumulative_map = self.cumulative_map_backward(energy_map)
            seam_idx = self.find_seam(cumulative_map)
            seams_record.append(seam_idx)
            self.delete_seam(seam_idx)

        self.out_image = np.copy(temp_image)
        n = len(seams_record)
        for dummy in range(n):
            seam = seams_record.pop(0)
            self.add_seam(seam)
            seams_record = self.update_seams(seams_record, seam)

    #计算能量矩阵，能量矩阵中使用了Scharr算子来检测图像的边缘，能量矩阵中能量越高的像素点表示其越重要，不能够被删除
    def calc_energy_map(self):
        b, g, r = cv2.split(self.out_image)
        b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
        g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
        r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
        return b_energy + g_energy + r_energy


    def cumulative_map_backward(self, energy_map):
        m, n = energy_map.shape
        output = np.copy(energy_map)
        for row in range(1, m):
            for col in range(n):
                output[row, col] = \
                    energy_map[row, col] + np.amin(output[row - 1, max(col - 1, 0): min(col + 2, n - 1)])
        return output

    # 
    def cumulative_map_forward(self, energy_map):
        matrix_x = self.calc_neighbor_matrix(self.kernel_x)
        matrix_y_left = self.calc_neighbor_matrix(self.kernel_y_left)
        matrix_y_right = self.calc_neighbor_matrix(self.kernel_y_right)

        m, n = energy_map.shape
        output = np.copy(energy_map)

        # 动归：M(i,j)=e(i,j)+min⁡(M(i-1,j-1),M(i-1,j),M(i-1,j+1))
        #递归每一行每一列，计算每行(i, j)所有可能连通接缝的累计最小能量M
        for row in range(1, m):     
            for col in range(n):
                #如果是最左边一列
                if col == 0:
                    e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_right, e_up)
                #如果是最右边的一列
                elif col == n - 1:
                    e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_left, e_up)
                #如果是中间的部分
                else:
                    e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                    e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_left, e_right, e_up)
        return output


    def calc_neighbor_matrix(self, kernel):
        b, g, r = cv2.split(self.out_image)
        output = np.absolute(cv2.filter2D(b, -1, kernel=kernel)) + \
                 np.absolute(cv2.filter2D(g, -1, kernel=kernel)) + \
                 np.absolute(cv2.filter2D(r, -1, kernel=kernel))
        return output

    #寻找符合条件的缝隙
    def find_seam(self, cumulative_map):
        m, n = cumulative_map.shape
        output = np.zeros((m,), dtype=np.uint32)
        output[-1] = np.argmin(cumulative_map[-1]) #在最后一行（即图像底部）寻找累积能量值最小的像素点，并保存该点的列索引值为 output[-1]。

        # 从倒数第二行开始，依次向上扫描每一行。对于每一行，都根据下一行中与上一个像素点和当前像素点相邻的三个像素的累积能量值，
        # 选择其中最小的一个像素点作为当前像素点所在的缝隙位置，并将该点的列索引值保存到结果数组 output 中。
        # 这段代码的作用是根据下一行已经选取的像素点的列索引值，在当前行中选取一个符合条件的像素点，
        # 并将其列索引值保存到结果数组 output 中，用于最终生成整幅图像的缝隙路径。
        for row in range(m - 2, -1, -1):
            prv_x = output[row + 1]
            if prv_x == 0:
                output[row] = np.argmin(cumulative_map[row, : 2])
            else:
                output[row] = np.argmin(cumulative_map[row, prv_x - 1: min(prv_x + 2, n - 1)]) + prv_x - 1
        return output

    #删除缝隙
    def delete_seam(self, seam_idx):
        m, n = self.out_image.shape[: 2]
        output = np.zeros((m, n - 1, 3))
        for row in range(m):
            col = seam_idx[row]
            output[row, :, 0] = np.delete(self.out_image[row, :, 0], [col])
            output[row, :, 1] = np.delete(self.out_image[row, :, 1], [col])
            output[row, :, 2] = np.delete(self.out_image[row, :, 2], [col])
        self.out_image = np.copy(output)

    #添加缝隙
    def add_seam(self, seam_idx):
        m, n = self.out_image.shape[: 2]
        output = np.zeros((m, n + 1, 3))
        for row in range(m):
            col = seam_idx[row]
            for ch in range(3):
                if col == 0:
                    p = np.average(self.out_image[row, col: col + 2, ch])
                    output[row, col, ch] = self.out_image[row, col, ch]
                    output[row, col + 1, ch] = p
                    output[row, col + 1:, ch] = self.out_image[row, col:, ch]
                else:
                    p = np.average(self.out_image[row, col - 1: col + 1, ch])
                    output[row, : col, ch] = self.out_image[row, : col, ch]
                    output[row, col, ch] = p
                    output[row, col + 1:, ch] = self.out_image[row, col:, ch]
        self.out_image = np.copy(output)


    def update_seams(self, remaining_seams, current_seam):
        output = []
        for seam in remaining_seams:
            seam[np.where(seam >= current_seam)] += 2
            output.append(seam)
        return output

    # 将图片旋转90度
    def rotate_image(self, image, ccw):
        m, n, ch = image.shape
        output = np.zeros((n, m, ch))
        if ccw:
            image_flip = np.fliplr(image)
            for c in range(ch):
                for row in range(m):
                    output[:, row, c] = image_flip[row, :, c]
        else:
            for c in range(ch):
                for row in range(m):
                    output[:, m - 1 - row, c] = image[row, :, c]
        return output

    #保存图像的最终结果到指定地址
    def save_result(self, filename):
        cv2.imwrite(filename, self.out_image.astype(np.uint8))

def image_resize_without_mask(filename_input, filename_output, height_percent_resize, width_percent_resize):
    obj = SeamCarver(filename_input, height_percent_resize, width_percent_resize)
    obj.save_result(filename_output)

if __name__ == '__main__':
    #指定缩放后图片的高度和宽度
    new_height = 312
    new_width = 312

    #指定期望的resize之后图片的长和宽占原来的百分比
    height_percent_resize=1
    width_percent_resize=0.6

    input_image="example/image.jpg"            #需要进行缩放的图片路径（相对路径）
    output_image="example/image_result3.jpg"   #处理后图片的输出路径
    #输出图片路径，检查是否正确
    print(input_image)
    print(output_image)
    image_resize_without_mask(input_image, output_image,height_percent_resize, width_percent_resize)