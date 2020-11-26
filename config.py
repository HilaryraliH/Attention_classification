'''通用的config，
还未考虑的功能：分支模型、滤波数据读取
'''

import os
# Covert数据中的1和6电极要去除
Cvert_all_elec = [i for i in range(62)]
Cvert_all_elec.remove(1)
Cvert_all_elec.remove(6)

Covert1s_elec = {
    'All': Cvert_all_elec,
    'EOG': [1, 6],
    'optimal9': [16, 24, 54, 55, 57, 58, 59, 60, 61],
    'F': [3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'C': [17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
    'P': [2, 5, 16, 24, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61],
    'Pleft': [16, 44, 45, 46, 47, 48, 54, 55, 59],
    'Pmid':  [48, 56, 60, 47, 55, 59, 49, 57, 61],
    'Pright': [24, 49, 50, 51, 52, 57, 58, 60, 61],
    'F9': [3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 54, 55, 57, 58, 59, 60, 61],
    'FC9': [3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 25,
              26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 16, 24, 54, 55, 57, 58, 59, 60, 61]
}


Covert2s_elec = Covert1s_elec


SimulEEG_elec = {
    'All': [i for i in range(28)],
    'F': [0,  2,  1,  3, 16, 17, 18],
    'P': [11, 12, 13, 14, 15, 25, 26, 27]
}

# t-sne画图的层数在模型的第几层
visual_layer_dict = {
    'JnecnnOrigin':-3,
    'EegnetOrigin':-3,
    'DeepconvnetOrigin':-3,
    'DeepconvnetOriginConstraint':-3,
    'ShallowconvnetOrigin':-3,
    'Base3D':-4,
    'Deep3D':-4,
    'Deep3DConstraint':-4,
    'Deep3DTwoBranch':-4,
    'Deep3DThreeBranch':-4,
    'Deep3DSmallKernel':-4,
    'JnecnnOriginConstraint':-3,
    'Deep3DTwoBranchResnet':-4,
    'Deep3DBatchNorm':-4
}

class config():
    def __init__(
        # 从 datasets 中选择数据集, 从 Covert_elec 和 SimulEEG_elec 中选择电极
        self, dt_idx, elec_area, 
        # 设定数据的输入方式： '2D' or '3D' ( 只有当 elec_area 为 All时，才能为 3D )
        dt_fm, 
        # 从 model 文件中选择模型
        mdl_nm, 
        # 若为 '3D'， 数据是否插值: True or False，并设置插值方式: 'linear', '2_poly', '3_poly'
        is_interpolate = False, interpolate_way = '3D_average', inter_size = 9, 
        # 每次交叉验证用 sub_num_each_cross 个被试，模型共跑cycles次
        sub_num_each_cross=1, cycles=1, 
        epochs=30, batch_size=32, optimizer='adam', 
        # 开始的cycle 和 开始的cross
        st_cycle = 0, st_cross = 0, need_cfg_2 = False,
        # 自己写创建的文件夹备注信息，3D模型是否插值，插值方法等，都要在这里表明
        other_info='', 
        # 是否进行减少数据的调试
        rdc_smp=False, rdc_smp_num=100
    ):

        # 一些既定的性质
        self.datasets = ['Covert1s', 'Covert2s', 'SimulEEG']

        # 从 datasets 中选择数据集
        self.dt_idx = dt_idx

        # 从 Covert_elec 和 SimulEEG_elec 中选择电极
        self.elec_area = elec_area

        # 设定数据的输入方式： '2D' or '3D'
        self.dt_fm = dt_fm

        # 若为 '3D'， 数据是否插值: True or False，并设置插值方式: 'linear', '2_poly', '3_poly'
        self.is_interpolate = is_interpolate
        self.interpolate_way = interpolate_way  # 'linear', 'nearest', 'cubic'
        self.inter_size = inter_size

        # 从 model 文件中选择模型
        self.mdl_nm = mdl_nm
        self.sub_num_each_cross = sub_num_each_cross
        self.cycles = cycles+1  # 每个模型只跑一次
        self.st_cycle = st_cycle
        self.st_cross = st_cross
        self.need_cfg_2 = need_cfg_2

        # 训练模型的参数
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.other_info = other_info  # 自己写创建的文件夹备注信息

        # 是否进行减少数据的调试
        self.rdc_smp = rdc_smp
        self.rdc_smp_num = rdc_smp_num

        # 一些不需要调的参数,可以算出来的
        self.dt_points = [200, 400, 400]
        self.dt_sub = [8, 8, 26]
        self.dt = self.datasets[self.dt_idx]  # 数据集名字
        self.sam_pnts = self.dt_points[self.dt_idx]  # 数据长度
        self.sub_num = self.dt_sub[self.dt_idx]  # 共 多少 个被试
        self.cross_num = self.sub_num//self.sub_num_each_cross # 共 多少 折交叉验证 （可算出每一个cross几个被试）
        self.elec = eval(self.dt+'_elec')[self.elec_area]  # 具体电极导联
        self.visual_layer = visual_layer_dict[self.mdl_nm]

    def set_dir(self, cycle, cross):
        # 设置数据读取路径
        self.dt_root_dir = '.'+os.sep+'Datasets'+os.sep
        self.dataset_path = self.dt_root_dir+'{}.mat'.format(self.dt)
        # self.dataset_path = self.dt_root_dir+'TestDataCell_62.mat' # 当用老师的数据进行对比时

        # 设置 处理后的数据 保存的文件夹
        self.process_dt_dir = self.dt_root_dir+'preprocess_data'+os.sep
        check_path(self.process_dt_dir)
        self.process_other_info = ''
        if self.is_interpolate:
            self.process_other_info = '_'+self.interpolate_way
        self.process_dt_dir = self.process_dt_dir + '{}_{}_{}{}'.format(self.dt, self.elec_area, self.dt_fm,self.process_other_info)+os.sep
        check_path(self.process_dt_dir)

        # 保存一个模型的 整个文件夹
        # 命名基础格式： 数据名 + 电极选择 + 模型名字 + 几折交叉 -- 是否插值 -- 其他信息
        self.root_dir = 'Result'+os.sep + str(self.dt)+'_'+self.elec_area + '_' + self.mdl_nm + '_' + '{}折'.format(self.cross_num)

        if self.is_interpolate:
            self.interpolate_info = '--' + self.interpolate_way # 将3D插值的信息，放在文件夹命名最后部分
            self.root_dir = self.root_dir+self.interpolate_info
        
        if self.other_info == '':
            self.root_dir = self.root_dir+os.sep
        else:
            self.root_dir = self.root_dir+'---------------'+self.other_info +os.sep
        check_path(self.root_dir)

        # 保存本cycle的 一次运行结果, 共 cycles 次
        self.save_dir = self.root_dir + '第{}次cycle'.format(cycle) +os.sep
        check_path(self.save_dir)

        # 保存 每次 cross 中，模型的名字
        self.save_mdl_dir = self.save_dir + 'save_model'+os.sep
        check_path(self.save_mdl_dir)
        self.save_mdl_file = self.save_mdl_dir + str(cross) + '.h5'

        # 保存 每次 cross 中，acc 趋势图
        self.save_acc_dir = self.save_dir + 'acc_trend'+os.sep
        check_path(self.save_acc_dir)
        self.save_acc_file = self.save_acc_dir + str(cross) + '.png'

        # 保存 每次 cross 中，loss 趋势图
        self.save_loss_dir = self.save_dir + 'loss_trend'+os.sep
        check_path(self.save_loss_dir)
        self.save_loss_file = self.save_loss_dir + str(cross) + '.png'


        # 保存 每次 cross 中，训练的具体信息：'val_loss', 'val_accuracy', 'loss', 'accuracy', 'lr'
        self.save_tr_process_dir = self.save_dir + 'tr_process'+os.sep
        check_path(self.save_tr_process_dir)
        self.save_tr_process_file = self.save_tr_process_dir + str(cross) + '.pkl'

        # 保存 每次 cross 中，对已有模型进行最后dense层的可分性的可视化 （t-sne）
        self.save_tsne_fig_dir = self.save_dir + 'dense_tsne_fig'+os.sep
        check_path(self.save_tsne_fig_dir)
        self.save_tsne_fig_file = self.save_tsne_fig_dir + str(cross) + '.png'

        # 将26张loss和acc训练趋势图保存在一张图上
        self.save_one_loss_file = self.save_loss_dir + 'total_loss.png'
        self.save_one_acc_file = self.save_acc_dir + 'total_acc.png'
        self.save_one_dense_file = self.save_tsne_fig_dir + 'total_dense_separable.png'

        # 保存本 cycle 的所有 混淆矩阵
        self.cnf_mtr_cycle_file = self.save_dir + 'cnf_mtr_cycle.npy'

        # 保存本 cycle 的所有指标
        self.acc_cycle_file = self.save_dir + 'acc_cycle.npy'
        self.best_acc_cycle_file = self.save_dir + 'best_acc_cycle.npy'
        self.pre_cycle_file = self.save_dir + 'pre_cycle.npy'
        self.re_cycle_file = self.save_dir + 're_cycle.npy'
        self.f1_cycle_file = self.save_dir + 'f1_cycle.npy'

        # 保存多次 cycle 的混淆矩阵, 在一张表内（每一列矩阵为一次cycle，若模型运行5次，则有5列）
        self.All_cnf_mtr_file = self.root_dir + 'All_cnf_mtr.npy'

        # 保存学习率变化的趋势图
        self.lr_change_file = self.save_dir + 'lr_changes.png'

        # 保存其它信息
        self.other_info_file = self.save_dir + 'other_info.csv'


def check_path(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except:
            print('make dir error')
            return
