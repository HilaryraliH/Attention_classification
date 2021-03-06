# coding=utf-8
import numpy as np
import pickle
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from config import config
from preprocess_data import load_data
from model import load_mdl_file


def plot_acc_for_each_mdl(acc_array)
    '''将 n_model*n_sub 的 acc 数组画出来，共 n_sub 张图，每张图 n_model 个点
    
    ----------
    Arguments:
    ----------
    acc_array：ndarray，shape = n_model*n_sub
    
    ----------                                              
    Returns:
    ----------
    An array of shape (N,H,W), which N denote the number of short sample we got
    '''


def my_append_row(total, one):
    '''连接每一个 cross 的混淆矩阵 (这一个 cycle 的混淆矩阵)，竖向连接，即增加行数'''
    if total is None:
        total = np.array(one)
    else:
        total = np.append(np.array(total), one, axis=0)
    return total


def my_append_col(total, one):
    '''连接每一个 cycle 的混淆矩阵，横向连接，即增加列数'''
    if total is None:
        total = np.array(one)
    else:
        total = np.append(np.array(total), one, axis=1)
    return total


def save_other_info(info_dict, cfg):
    '''将字典的 键值对 存到表格文件'''

    f = open(cfg.other_info_file, 'w')  # 以'w'方式打开文件
    for k, v in info_dict.items():  # 遍历字典中的键值
        s = str(v)  # 把字典的值转换成字符型
        f.write(k + ',' + s + '\n')  # 键和值放在一行，以回车结束,其中的逗号是区分csv文件中的单元格的
    f.write('epochs,' + str(cfg.epochs) + '\n')
    f.write('batch_size:,' + str(cfg.batch_size) + '\n')
    f.close()


def plot_dense_embed(mdl,cfg, nrow,ncol,val_x,val_y,cross):
    '''画每个sub的可分性，放到一张大画布中的相应位置（第sub个位置）'''
    # 取出模型的第一层和dense层，dense层作为输出，构造新的模型
    out_ly_shape = mdl.layers[cfg.visual_layer].output
    act_mdl = Model(inputs=mdl.input, outputs=out_ly_shape)

    # 对验证数据进行验证，
    dt = act_mdl.predict(val_x)

    # 构造tsne模型
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    # 将数据降维
    tans_dt = tsne.fit_transform(dt)
    # 得到需要画的数据
    y = np.where(val_y==1)[1]
    idx_0 = np.where(y==0)
    idx_1 = np.where(y==1)
    x1 = tans_dt[idx_0,0]
    x2 = tans_dt[idx_1,0]
    y1 = tans_dt[idx_0,1]
    y2 = tans_dt[idx_1,1]

    # 根据label画出相应的颜色
    # 对于SimulEEG，att label 为 0，non-att label 为 1
    # 对于covert，att label 为 1，non-att label 为 0
    plt.subplot(nrow,ncol,cross+1)
    if cfg.dt=='SimulEEG':
        plt.scatter(x1,y1,c='red',marker='.',label='att')
        plt.scatter(x2,y2,c='blue',marker='.',label='non_att')
    else:
        plt.scatter(x1,y1,c='blue',marker='.',label='non_att')
        plt.scatter(x2,y2,c='red',marker='.',label='att')
    plt.title(str(cross))
    plt.axis('off')


def plot_acc_embed(hist, nrow,ncol,cross):
    '''画每个sub的acc小图，放到一张大画布中的相应位置（第sub个位置）'''

    # 保存 acc 趋势图
    metric = 'accuracy'
    plt.subplot(nrow,ncol,cross+1)
    plt.plot(hist[metric])
    plt.plot(hist['val_' + metric])
    plt.axis('off')


def plot_loss_embed(hist, nrow,ncol,cross):
    '''画每个sub的loss小图，放到一张大画布中的相应位置（第sub个位置）'''
    # 保存 loss 趋势图
    metric = 'loss'
    plt.subplot(nrow,ncol,cross+1) # 在画布的第 cross+1 个位置画
    plt.plot(hist[metric])
    plt.plot(hist['val_' + metric])
    plt.axis('off')


def plot_dense_big(cfg,nrow,ncol, st_cycle=1,st_cross=1):
    '''将可分性画在一张大画布上，共 sub_num 个小图，共 nrow*ncol 个格子'''
    for cycle in range(st_cycle, cfg.cycles+1):
        # 建立一张大画布
        plt.figure()

        for cross in range(st_cross, cfg.cross_num+1):
            cfg.set_dir(cycle, cross)

            # 读取数据
            print('\n', '--'*20, 'Start load data from dataset {} '.format(cfg.dt), '--'*20, '\n')
            __, __, val_x, val_y = load_data(cross, cfg,get_tr=False) # 若cfg_2不为None，则返回两种数据的列表

            # load 模型, 并显示结构
            mdl = load_mdl_file(cfg)
            if cross==0 and cycle==0:
                mdl.summary()

            # 画出模型在 dense 层的可分性，再保存图片
            plot_dense_embed(mdl, cfg, nrow, ncol, val_x, val_y, cross)

        # 保存大画布到文件
        plt.suptitle(cfg.mdl_nm+'__'+'cycle'+str(cycle)+'__'+'dense_separability')
        plt.savefig(cfg.save_one_dense_file, bbox_inches='tight')
        plt.close()


def plot_acc_or_loss_big(cfg,metrics,nrow,ncol, st_cycle=1,st_cross=1):
    '''将 acc 或者 loss 画在一张大画布上，共 sub_num 个小图，共 nrow*ncol 个格子'''
    for cycle in range(st_cycle,cfg.cycles+1):
        # 建立一张大画布
        plt.figure()

        for cross in range(st_cross,cfg.cross_num+1):
            cfg.set_dir(cycle,cross)
            
            # load history 文件
            df=open(cfg.save_tr_process_file,'rb')#注意此处是rb
            hist=pickle.load(df)
            df.close() 

            # 保存训练时的 acc、loss 趋势图，并保存到一张大图上
            if metrics=='accuracy':
                plot_acc_embed(hist,nrow,ncol,cross)
            elif metrics=='loss':
                plot_loss_embed(hist,nrow,ncol,cross)
        

        # 保存大画布到文件
        plt.suptitle(cfg.mdl_nm+metrics)
        if metrics=='accuracy':
            plt.savefig(cfg.save_one_acc_file, bbox_inches='tight')
        elif metrics=='loss':
            plt.savefig(cfg.save_one_loss_file, bbox_inches='tight')
        plt.close()


def plot_dense_single(mdl,cfg,val_x,val_y,cross):
    '''单独画出模型在 dense 层的可分性，单独保存为一张图片'''
    # 取出模型的第一层和dense层，dense层作为输出，构造新的模型
    out_ly_shape = mdl.layers[cfg.visual_layer].output
    act_mdl = Model(inputs=mdl.input, outputs=out_ly_shape)

    # 对验证数据进行验证，
    dt = act_mdl.predict(val_x)

    # 构造tsne模型
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    # 将数据降维
    tans_dt = tsne.fit_transform(dt)
    # 得到需要画的数据
    y = np.where(val_y==1)[1]
    idx_0 = np.where(y==0)
    idx_1 = np.where(y==1)
    x1 = tans_dt[idx_0,0]
    x2 = tans_dt[idx_1,0]
    y1 = tans_dt[idx_0,1]
    y2 = tans_dt[idx_1,1]

    # 根据label画出相应的颜色
    # 对于SimulEEG，att label 为 0，non-att label 为 1
    # 对于covert，att label 为 1，non-att label 为 0
    plt.figure()
    if cfg.dt=='SimulEEG':
        plt.scatter(x1,y1,c='red',marker='.',label='att')
        plt.scatter(x2,y2,c='blue',marker='.',label='non_att')
    else:
        plt.scatter(x1,y1,c='blue',marker='.',label='non_att')
        plt.scatter(x2,y2,c='red',marker='.',label='att')
    plt.title(cfg.mdl_nm+'__'+'sub'+str(cross)+'__'+'dense_separability')
    plt.legend()
    plt.savefig(cfg.save_tsne_fig_file)
    print('saved figure {}'.format(cfg.save_tsne_fig_file))


def plot_acc_or_loss_single(hist, cfg):
    '''单独画出 acc 或 loss，单独保存为一张图片'''

    # 保存 acc 趋势图
    plt.figure()
    metric = 'accuracy'
    plt.plot(hist[metric])
    plt.plot(hist['val_' + metric])
    plt.title('val_' + metric + ':' + str(hist['val_' + metric][-1]))
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(cfg.save_acc_file, bbox_inches='tight')
    plt.close()

    # 保存 loss 趋势图
    plt.figure()
    metric = 'loss'
    plt.plot(hist[metric])
    plt.plot(hist['val_' + metric])
    plt.title('val_' + metric + ':' + str(hist['val_' + metric][-1]))
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(cfg.save_loss_file, bbox_inches='tight')
    plt.close()


def plot_dense_all_sigle(cfg, st_cycle=1,st_cross=1):
    '''直接画出所有的 dense 可分性图片，不在main.py中调用
    在没有这个图的时候，单独运行这个文件来补画
    '''
    for cycle in range(st_cycle,cfg.cycles+1):
        for cross in range(st_cross,cfg.cross_num+1):
            cfg.set_dir(cycle,cross)

            # 读取数据
            print('\n', '--'*20, 'Start load data from dataset {} '.format(cfg.dt), '--'*20, '\n')
            __, __, val_x, val_y = load_data(cross, cfg,get_tr=False) # 若cfg_2不为None，则返回两种数据的列表

            # load 模型, 并显示结构
            mdl = load_mdl_file(cfg)
            if cross==0 and cycle==0:
                mdl.summary()

            # 画出模型在 dense 层的可分性，再保存图片
            plot_dense_single(mdl,cfg,val_x,val_y,cross)

# 单独运行此文件时，要调的参数都在  if __name__ == "__main__" 内部

if __name__ == "__main__":
    # 若main.py中没有正确地进行画图，则运行此文件重新画图
    # 要注意cfg的设置
    st_cycle = 1
    st_cross = 1
    need_cfg_2 = False
    plot_all_dense = False
    plot_one_dense = True
    plot_one_acc = True
    plot_one_loss = True
    nrow = 5
    ncol = 6

    cfg = config(
        # 从 datasets 中选择数据集 datasets = ['Covert1s','Covert2s', 'SimulEEG']
        dt_idx = 2,
        # 从 Covert_elec 和 SimulEEG_elec 中选择电极
        elec_area = 'All',
        # 设定数据的输入方式： '2D' or '3D' ( 只有当 elec_area 为 All时，才能为 3D )
        dt_fm = '3D',
        is_Z_Norm = False,
        # 从 model 文件中选择模型
        mdl_nm = 'JnecnnOrigin',
        # 每个模型跑几次
        cycles = 1, 
        # 开始的cycle 和 开始的cross
        st_cycle = 1,
        st_cross = 1,
        # 自己写创建的文件夹备注信息，3D模型是否插值，插值方法等，都要在这里表明
        other_info = '32-32-64'
    )

    if plot_all_dense:
        plot_dense_all_sigle(cfg)
    if plot_one_dense:
        plot_dense_big(cfg,nrow,ncol)
    if plot_one_acc:
        plot_acc_or_loss_big(cfg,'accuracy',nrow,ncol)
    if plot_one_loss:
        plot_acc_or_loss_big(cfg,'loss',nrow,ncol)




