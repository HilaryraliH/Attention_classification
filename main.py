# 系统库
import os
import pickle
import numpy as np
from time import time
import tensorflow as tf
from keras.utils import plot_model

# 自己文件
from config import config
from preprocess_data import load_data
from visualization import plot_acc_or_loss_single,plot_dense_single
from model import blt_mdl, eval_mdl, fit_mdl,load_mdl_file
from save_info import save_other_info,my_append_row,my_append_col

# 不全部占满显存, 按需分配
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_config = tf.compat.v1.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=gpu_config)
tf.compat.v1.keras.backend.set_session(sess)

# 画出模型所需要的包，配入环境变量
os.environ["PATH"] += os.pathsep + 'C:/C1_Install_package/Graphviz/Graphviz 2.44.1/bin'

# 设置配置参数,并指定从哪一折开始训练
cfg = config(
    # 从 datasets 中选择数据集 datasets = ['Covert1s','Covert2s', 'SimulEEG']
    dt_idx = 2,
    # 从 Covert_elec 和 SimulEEG_elec 中选择电极
    elec_area = 'All',
    # 设定数据的输入方式： '2D' or '3D' ( 只有当 elec_area 为 All时，才能为 3D )
    dt_fm = '3D',
    # 从 model 文件中选择模型
    mdl_nm = 'Deep3DTwoBranchBigResnet',
    # 每个模型跑几次, 只关心几次，不要关心是从0还是从1开始
    cycles = 2,
    # 开始的cycle 和 开始的cross，此处的概念都是从1开始
    st_cycle = 1,
    st_cross = 1,
    # 自己写创建的文件夹备注信息，3D模型是否插值，插值方法等，都要在这里表明
    other_info = '32-32-64'
)


# cfg_2是用作对第二种数据形式的提取设定，对模型的设置可以不用管
# 并且注意，除了两者数据的方式 2D 和 3D 不一样之外，其他设置都相同
# 若不需要，则将下面的cfg_2=None
# 主要模型设置还是要看 cfg
cfg_2 = None
if cfg.need_cfg_2:
    cfg_2 = config(
        # 从 datasets 中选择数据集
        dt_idx = 2,
        # 从 Covert_elec 和 SimulEEG_elec 中选择电极
        elec_area = 'All',
        # 设定数据的输入方式： '2D' or '3D' ( 只有当 elec_area 为 All时，才能为 3D )
        dt_fm = '3D',
        # 若为 '3D'， 数据是否插值: True or False，并设置插值方式: 'linear', '2_poly', '3_poly'
        is_interpolate = False,
        interpolate_way = '3D_average',  # 'linear', 'nearest', 'cubic', 'average','3D_average'
        inter_size = 9,
        # 从 model 文件中选择模型
        mdl_nm = 'Deep3DThreeBranch',
        sub_num_each_cross = 1,
        cycles = 1,  # 每个模型只跑一次
        epochs = 30,
        batch_size = 32,
        optimizer = 'adam',
        other_info = '第一个block为32个kernel',  # 自己写创建的文件夹备注信息，3D模型是否插值，插值方法等，都要在这里表明
        # 是否进行减少数据的调试
        rdc_smp = False,
        rdc_smp_num = 100
    )



# 所有的混淆矩阵
All_cnf_mtr = None

for cycle in range(cfg.st_cycle,cfg.cycles):

    # 定义保存所有cross指标的变量
    start = time()
    acc_list = []
    best_acc_list = []
    f1_list = []
    pre_list = []
    re_list = []
    cnf_mtr_cycle = None  # use to save_to_csv
    numParams = None
    info_dict = {}


    for cross in range(cfg.st_cross,cfg.cross_num):

        # 定义保存一个cross指标的变量
        cfg.set_dir(cycle, cross)
        if cfg_2:
            cfg_2.set_dir(cycle, cross)
        cnf_mtr_cross = None
        acc = None
        mdl = None

        # 若程序中断过，则存在这些结果文件，则将每一个都load进来继续训练，但其中的训练时间无法更新，所以，对于时间，看看就好
        # 但这段程序，也让除了第0次，之后的每次训练，都会从前面的文件中load变量
        print("\n\n\n",'#' * 70,"Start process of {} cross ".format(cross),'#' * 70)

        if os.path.exists(cfg.cnf_mtr_cycle_file):
            cnf_mtr_cycle = np.load(cfg.cnf_mtr_cycle_file)
            acc_list = np.load(cfg.acc_cycle_file)
            best_acc_list = np.load(cfg.best_acc_cycle_file)
            pre_list = np.load(cfg.pre_cycle_file)
            re_list = np.load(cfg.re_cycle_file)
            f1_list = np.load(cfg.f1_cycle_file)
            print("\nReloading results from file...... \n")


        # 读取数据
        print('\n', '--'*20, 'Start load data from dataset {} '.format(cfg.dt), '--'*20, '\n')
        tr_x, tr_y, val_x, val_y = load_data(cross, cfg, cfg_2) # 若cfg_2不为None，则返回两种数据的列表


        # 若需要调试，则减少数据
        if cfg.rdc_smp:
            tr_x = tr_x[:cfg.rdc_smp_num]
            tr_y = tr_y[:cfg.rdc_smp_num]
            val_x = val_x[:cfg.rdc_smp_num]
            val_y = val_y[:cfg.rdc_smp_num]

        # 建立模型
        if os.path.exists(cfg.save_mdl_file):
            # 若存在之前训练过的模型，则直接load
            mdl = load_mdl_file(cfg)
        else:
            # 建立单个模型，而分支模型在model.py中则定义为了一个模型，也即cfg中的模型名字
            mdl = blt_mdl(cfg,cfg_2)


        # 窗口中显示模型 或者 画出模型
        if cross == 0 and cycle == 0:
            print('\n', '--'*20, 'Start plot mdl image', '--'*20, '\n')
            mdl.summary()
            # plot_model(mdl, to_file=cfg.root_dir + cfg.mdl_nm+'.png', show_shapes=True)


        # 训练模型
        print('\n', '--'*20, 'start training {}th fold'.format(cross), '--'*20, '\n')
        if os.path.exists(cfg.save_mdl_file):

            # 若存在模型，则将前面load的模型直接在验证集上验证结果
            print('Model already exists, evaluate acc from {} ......'.format(cfg.save_mdl_file))
            acc, cnf_mtr_cross = eval_mdl(mdl, val_x, val_y)
        else:

            # 从头开始训练模型
            hist = fit_mdl(mdl, cfg, tr_x, tr_y, val_x, val_y)

            # 保存训练过程中时，各种具体信息
            file = open(cfg.save_tr_process_file, 'wb')
            pickle.dump(hist.history, file)
            file.close()

            # 保存训练时的acc趋势图
            plot_acc_or_loss_single(hist.history, cfg)

            # 用验证集验证模型
            acc, cnf_mtr_cross = eval_mdl( mdl, val_x, val_y)

            # 保存最佳epoch acc的结果
            best_acc = np.max(hist.history['val_accuracy'])
            best_acc_list = np.append(best_acc_list, best_acc)


            # 读取pkl文件举例
            # file = open(cfg.save_tr_process_file, 'rb')
            # loaded_data = pickle.load(file, encoding='bytes')
            # loaded_data.keys()

        # 画出模型在 dense 层的可分性，再保存图片
        plot_dense_single(mdl,cfg,val_x,val_y,cross)

        # 保存参数数量
        info_dict['numParams'] = mdl.count_params()

        # 保存每一个 cross 的混淆矩阵 (这一个 cycle 的混淆矩阵)
        cnf_mtr_cycle = my_append_row(cnf_mtr_cycle, cnf_mtr_cross)

        # 保存每一个 cross 的 指标 (这一个 cycle 的 指标 )
        pre = cnf_mtr_cross[0,0]/(cnf_mtr_cross[0,0]+cnf_mtr_cross[1,0])
        re = cnf_mtr_cross[0,0]/(cnf_mtr_cross[0,0]+cnf_mtr_cross[0,1])
        f1 = 2*pre*re/(pre+re)
        acc_list = np.append(acc_list, acc)  # save each sub's acc
        pre_list = np.append(pre_list, pre)
        re_list = np.append(re_list, re)
        f1_list = np.append(f1_list, f1)

        # 为防止程序中断，每一个 cross 都要保存到文件一次
        np.save(cfg.cnf_mtr_cycle_file ,cnf_mtr_cycle)
        np.save(cfg.acc_cycle_file, np.array(acc_list))
        np.save(cfg.best_acc_cycle_file, np.array(best_acc_list))
        np.save(cfg.pre_cycle_file, np.array(pre_list))
        np.save(cfg.re_cycle_file, np.array(re_list))
        np.save(cfg.f1_cycle_file, np.array(f1_list))
        print("\nClassification accuracy of {} cross: {} ".format(cross,acc))

        # 清除内存
        del tr_x, tr_y, val_x, val_y
        import gc
        gc.collect()

    # 跨 cycle 保存文件

    # 计算程序运行时间
    end = time()
    info_dict['training_time'] = str((end - start) / 3600) + ' hours'

    # 保存各种指标的 平均值 和 标准差
    info_dict['acc aver'] = np.mean(acc_list)*100
    info_dict['best acc aver'] = np.mean(best_acc_list)*100
    info_dict['pre aver'] = np.mean(pre_list)*100
    info_dict['re aver'] = np.mean(re_list)*100
    info_dict['f1 aver'] = np.mean(f1_list)*100

    info_dict['acc std'] = np.std(acc_list)*100
    info_dict['best acc std'] = np.std(best_acc_list)*100
    info_dict['pre std'] = np.std(pre_list)*100
    info_dict['re std'] = np.std(re_list)*100
    info_dict['f1 std'] = np.std(f1_list)*100


    # 保存 info_dict
    save_other_info(info_dict, cfg)

    # 保存多次 cycle 的混淆矩阵, 防止中断，每一个cycle都要保存
    All_cnf_mtr = my_append_col(All_cnf_mtr, cnf_mtr_cycle)
    np.save(cfg.All_cnf_mtr_file, All_cnf_mtr)



