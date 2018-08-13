import struct
import numpy as np
import argparse
import os
from sklearn.preprocessing import scale



"""
THE IDX FILE FORMAT

the IDX file format is a simple format for vectors and multidimensional matrices of various numerical types.
The basic format is

magic number
size in dimension 0
size in dimension 1
size in dimension 2
.....
size in dimension N
data

The magic number is an integer (MSB first). The first 2 bytes are always 0.

The third byte codes the type of the data:
0x08: unsigned byte
0x09: signed byte
0x0B: short (2 bytes)
0x0C: int (4 bytes)
0x0D: float (4 bytes)
0x0E: double (8 bytes)

The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....

The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).

The data is stored like in a C array, i.e. the index in the last dimension changes the fastest.

[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  60000            number of images 
0008     32 bit integer  28               number of rows 
0012     32 bit integer  28               number of columns 
0016     unsigned byte   ??               pixel 
0017     unsigned byte   ??               pixel 
........ 
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

Decode IDX formate and save b/w single channel images in a big matrix with shape (num,h,w)
"""
def decode_img(path):

    with open(path,'rb') as img_file:
        img_binary = img_file.read()
        offset = 0
        header = '>iiii' #MSB first, high endian
        _, num_images, h, w = struct.unpack_from(header, img_binary, offset)
        offset += struct.calcsize(header)
        img_formate = '>' + str(h*w) + 'B'# for unsigned int
        print("======training set===========")
        print(f"number of image:{num_images}")
        print(f"image size: {h} X {w}")
        print("=============================")
        imgs = np.zeros((num_images,1,h,w))
        for i in range(num_images):
            if (i + 1) % 1000 == 0:
                print("processing:{:>10.2%}".format(i/num_images))
            imgs[i,0] = scale(X=np.array(struct.unpack_from(img_formate, img_binary, offset))).reshape((h,w))
            #imgs[i, 0] = np.array(struct.unpack_from(img_formate, img_binary, offset)).reshape((h, w))
            offset += struct.calcsize(img_formate)
        return imgs

"""

[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
0004     32 bit integer  10000            number of items 
0008     unsigned byte   ??               label 
0009     unsigned byte   ??               label 
........ 
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.

Decode IDX formate label and save label in an array
"""

def decode_lable(path):
    print("Decoding label...")
    with open(path,'rb') as labels_file:
        labels_binary = labels_file.read()
        offset = 0
        header = '>ii'
        _,num = struct.unpack_from(header,labels_binary,offset)
        offset += struct.calcsize(header)
        labels_format = '>'+str(num)+'B'
        labels = np.array(struct.unpack_from(labels_format,labels_binary,offset))
        print('Done!')
        return labels

def main(args):
    images_path = args.i
    # 训练集标签文件
    labels_path = args.l
    # 测试集文件

    imgs_mat = decode_img(images_path)
    labels_array = decode_lable(labels_path)
    outpath = args.o if args.o[-1] != '/' else args.o[:-1]
    if os.path.exists(outpath):
        np.save(outpath+'/image.npy',imgs_mat)
        np.save(outpath+'/label.npy',labels_array)
    else:
        print("Output Path not exist, save in current workdir.")
        np.save(os.path.join(os.getcwd(),"image.npy"), imgs_mat)
        np.save(os.path.join(os.getcwd(),"labels.npy"), labels_array)



if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description='Usage: $python convert_mnist.py -i ImgsFilesPath -l labelsFilesPath [-o OutputDirPath]\n'
                    'Default output path is current workdir.')

    parse.add_argument("-i", required=True, help="Binary imgs files path.")
    parse.add_argument("-l", required=True, help="Binary labels files path.")
    parse.add_argument("-o", default=os.getcwd(), help="Output path, default is same as origin path of binary file.")
    args = parse.parse_args()
    main(args)