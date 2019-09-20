import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os 

# 可视化hdr文件，传入hdr的路径和存储目的文件夹位置即可
def visualize_dataset(hdr_path,path):
    data = nib.load(hdr_path).get_data()
    data = np.squeeze(data,axis=-1)
    if not os.path.exists(path):
        os.mkdir(path)
    for i,img in enumerate(data):
        # plt.imshow(img)
        # plt.savefig()
        plt.imsave(os.path.join(path,'%d.png'%i),img)
        print('figure i %d ok'%i)

# 遍历文件夹生成描述txt文件，废弃
def generate_trainval_txt(path,mode):
    files = os.listdir(path)
    T1_image,T2_image,gt = [],[],[]
    for file in files:
        if file.endswith('.hdr'):
            if file.find('T1') != -1:
                T1_image.append(file)
            elif file.find('T2') != -1:
                T2_image.append(file)
            elif file.find('label') != -1:
                gt.append(file)
            else:
                raise 'illegal image name'
    T1_image = sorted(T1_image)
    T2_image = sorted(T2_image)
    gt = sorted(gt)
    if mode == 'train':
        discription = open(os.path.join(path,'train.txt'),mode='w+')
    elif mode == 'val':
        discription = open(os.path.join(path,'val.txt'),mode='w+')
    else:
        raise 'only train and val mode supported'
    for i in range(len(T1_image)):
        discription.write('{} {} {}\n'.format(os.path.join(path,T1_image[i]),
                                              os.path.join(path,T2_image[i]),
                                              os.path.join(path,gt[i])))
    discription.close()
    return T1_image,T2_image,gt




if __name__ == '__main__':
    #generate_trainval_txt('./data/training',mode='train')
    #generate_trainval_txt('./data/val',mode='val')
    visualize_dataset('./data/test2/subject-28-T1.hdr','./tmp/T1_28')
    #visualize_dataset('data/test/subject-11-T1.hdr','./output/subject_11_T1')