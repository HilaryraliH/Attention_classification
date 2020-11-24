# 同步 2020-11-24

# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

def my_append_row(total, one):
    if total is None:
        total = np.array(one)
    else:
        total = np.append(np.array(total), one, axis=0)
    return total


def my_append_col(total, one):
    if total is None:
        total = np.array(one)
    else:
        total = np.append(np.array(total), one, axis=1)
    return total


def save_training_pic(hist, cfg):
    '''保存每一次训练的每一张图'''

    # 保存 acc 趋势图
    plt.figure()
    metric = 'accuracy'
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('val_' + metric + ':' + str(hist.history['val_' + metric][-1]))
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(cfg.save_acc_file, bbox_inches='tight')
    plt.close()

    # 保存 loss 趋势图
    plt.figure()
    metric = 'loss'
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('val_' + metric + ':' + str(hist.history['val_' + metric][-1]))
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(cfg.save_loss_file, bbox_inches='tight')
    plt.close()


def save_training_acc_pic_in_one_fig(hist, nrow,ncol,cross):
    '''保存每一次训练的每一张图'''

    # 保存 acc 趋势图
    metric = 'accuracy'
    plt.subplot(nrow,ncol,cross+1)
    plt.plot(hist[metric])
    plt.plot(hist['val_' + metric])
    plt.axis('off')




def save_training_loss_pic_in_one_fig(hist, nrow,ncol,cross):
    # 保存 loss 趋势图
    metric = 'loss'
    plt.subplot(nrow,ncol,cross+1) # 在画布的第 cross+1 个位置画
    plt.plot(hist[metric])
    plt.plot(hist['val_' + metric])
    plt.axis('off')




def save_other_info(info_dict, cfg):

    f = open(cfg.other_info_file, 'w')  # 以'w'方式打开文件
    for k, v in info_dict.items():  # 遍历字典中的键值
        s = str(v)  # 把字典的值转换成字符型
        f.write(k + ',' + s + '\n')  # 键和值放在一行，以回车结束,其中的逗号是区分csv文件中的单元格的
    f.write('epochs,' + str(cfg.epochs) + '\n')
    f.write('batch_size:,' + str(cfg.batch_size) + '\n')
    f.close()
