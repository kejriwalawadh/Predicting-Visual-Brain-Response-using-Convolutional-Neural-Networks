import cv2
import os
import glob
from matplotlib import pyplot as plt

def main():
    fig = plt.figure(figsize=(15,15))
    rows = 10
    columns = 10
    dir = os.path.join('92images')
    img_list = glob.glob(dir+'/*.jpg')

    for i in range(90):
        fig.add_subplot(rows, columns, i+1)
        img = cv2.imread(img_list[i])[:,:,::-1]
        plt.imshow(img, interpolation='nearest')
        plt.axis('off')

    fig.add_subplot(rows, columns, 95)
    img = cv2.imread(img_list[90])[:,:,::-1]
    plt.imshow(img, interpolation='nearest')
    plt.axis('off')

    fig.add_subplot(rows, columns, 96)
    img = cv2.imread(img_list[91])[:,:,::-1]
    plt.imshow(img, interpolation='nearest')
    plt.axis('off')

    plt.savefig('Image_grid.png')


if __name__ == '__main__':
    main()
