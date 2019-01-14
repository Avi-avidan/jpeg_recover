import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import os

def save_image(img, filedir, i, j=0, score=0, name=''):
    try:
        create_dir(filedir)
        dst_path = filedir + '/epoch_{}_{}_{}_{:.6f}.jpg'.format(i, j, name, score)
        save_img = np.asarray(np.clip((img + 1.)*127.5, 0., 255.), dtype=np.uint8)
        plt.imsave(dst_path, save_img)
    except:
        print('failed to save img:', dst_path)
        print('save img shape, max, min, type:', save_img.shape, np.max(save_img), np.min(save_img), save_img.dtype)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_one_of_each(img, label, name):
    f, ax = plt.subplots(1, 3, figsize=(16,8))
    ax[0].imshow(img)
    ax[1].imshow(label)
    ax[2].imshow(label-img, cmap='hot')
    ax[0].set_title(name)
    ax[1].set_title(name.replace('.jpg', '.bmp'))
    ax[2].set_title('diff')
    plt.show()
    return img, label


def get_img_stats(img, print_stats=False):
    img_max, img_min, img_mean, img_std = np.max(img), np.min(img), np.mean(img), np.std(img)
    if print_stats:
        print('max: {}, min: {}, mean: {}, std: {}'.format(img_max, img_min, img_mean, img_std))
    return img_max, img_min, img_mean, img_std
    

def plot_dif(img1, img2, print_stats=True, plot=False):
    #dif = np.uint8(img1) - np.uint8(img2)
    dif = img2 - img1
    if print_stats:
        get_img_stats(img1, print_stats=print_stats)
        get_img_stats(img2, print_stats=print_stats)
        get_img_stats(dif, print_stats=print_stats)
    if plot:
        plot_img(dif)
    return dif

def get_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))