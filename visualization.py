# 同步 2020-11-24

import numpy as np
import pickle
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from config import config
from preprocess_data import load_data
from model import load_mdl_file
from save_info import save_training_acc_pic_in_one_fig,save_training_loss_pic_in_one_fig

def plot_dense_separability(mdl,cfg,val_x,val_y,cross):
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


def plot_all_dense_separability_func(cfg, st_cycle=0,st_cross=0):
    for cycle in range(st_cycle,cfg.cycles):
        for cross in range(st_cross,cfg.cross_num):
            cfg.set_dir(cycle,cross)

            # 读取数据
            print('\n', '--'*20, 'Start load data from dataset {} '.format(cfg.dt), '--'*20, '\n')
            __, __, val_x, val_y = load_data(cross, cfg,get_tr=False) # 若cfg_2不为None，则返回两种数据的列表

            # load 模型, 并显示结构
            mdl = load_mdl_file(cfg)
            if cross==0 and cycle==0:
                mdl.summary()

            # 画出模型在 dense 层的可分性，再保存图片
            plot_dense_separability(mdl,cfg,val_x,val_y,cross)


def plot_one_fig_func(cfg,metrics,nrow,ncol, st_cycle=0,st_cross=0):

    for cycle in range(st_cycle,cfg.cycles):
        # 建立一张大画布
        plt.figure()

        for cross in range(st_cross,cfg.cross_num):
            cfg.set_dir(cycle,cross)
            
            # load history 文件
            df=open(cfg.save_tr_process_file,'rb')#注意此处是rb
            hist=pickle.load(df)
            df.close() 

            # 保存训练时的 acc、loss 趋势图，并保存到一张大图上
            if metrics=='accuracy':
                save_training_acc_pic_in_one_fig(hist,nrow,ncol,cross)
            elif metrics=='loss':
                save_training_loss_pic_in_one_fig(hist,nrow,ncol,cross)
        

        # 保存大画布到文件
        plt.suptitle(cfg.mdl_nm+metrics)
        if metrics=='accuracy':
            plt.savefig(cfg.save_one_acc_file, bbox_inches='tight')
        elif metrics=='loss':
            plt.savefig(cfg.save_one_loss_file, bbox_inches='tight')
        plt.close()

    
# 单独运行此文件时，要调的参数都在  if __name__ == "__main__" 内部

if __name__ == "__main__":
    # 若main.py中没有正确地进行画图，则运行此文件重新画图
    # 要注意cfg的设置
    st_cycle = 0
    st_cross = 0
    need_cfg_2 = False
    plot_dense = False
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
        # 从 model 文件中选择模型
        mdl_nm = 'Deep3DTwoBranch',
        # 每个模型跑几次
        cycles = 2, 
        # 开始的cycle 和 开始的cross
        st_cycle = 0,
        st_cross = 0,
        # 自己写创建的文件夹备注信息，3D模型是否插值，插值方法等，都要在这里表明
        other_info = '64-32-16'
    )

    if plot_dense:
        plot_all_dense_separability_func(cfg)
    if plot_one_acc:
        plot_one_fig_func(cfg,'accuracy',nrow,ncol)
    if plot_one_loss:
        plot_one_fig_func(cfg,'loss',nrow,ncol)



