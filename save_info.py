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






def save_other_info(info_dict, cfg):

    f = open(cfg.other_info_file, 'w')  # 以'w'方式打开文件
    for k, v in info_dict.items():  # 遍历字典中的键值
        s = str(v)  # 把字典的值转换成字符型
        f.write(k + ',' + s + '\n')  # 键和值放在一行，以回车结束,其中的逗号是区分csv文件中的单元格的
    f.write('epochs,' + str(cfg.epochs) + '\n')
    f.write('batch_size:,' + str(cfg.batch_size) + '\n')
    f.close()
