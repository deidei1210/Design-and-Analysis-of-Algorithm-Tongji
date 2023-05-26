from seam_carving import SeamCarver

import os



def image_resize_without_mask(filename_input, filename_output, new_height, new_width):
    obj = SeamCarver(filename_input, new_height, new_width)
    obj.save_result(filename_output)

if __name__ == '__main__':

    folder_in = 'in'
    folder_out = 'out'

    filename_input = 'image.png'
    filename_output = 'image_result.png'
    filename_mask = 'mask.jpg'
    new_height = 200
    new_width = 512

    input_image="example/image6.jpg"
    output_image="example/image6_result.jpg"
    print(input_image)
    # print(input_mask)
    print(output_image)
    image_resize_without_mask(input_image, output_image, new_height, new_width)
