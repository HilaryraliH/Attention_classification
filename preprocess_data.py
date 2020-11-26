# coding=utf-8
import os
import numpy as np
import scipy.io as sio
from keras.utils import to_categorical
from scipy.interpolate import griddata
from scipy import interpolate
import matplotlib as mpl
import pylab as pl


# 如果 print 需要彩色： \033[1;32;m{}\033[0m

def load_data(sub, cfg, cfg_2=None,get_tr=True,get_val=True):
    # 初始化数据
    tr_dt, tr_lab, val_dt, val_lab = None,None,None,None
    dt, lab = {}, {}

    # 构造 val_sub
    all_sub = [i for i in range(1,cfg.sub_num)]
    val_start_sub = sub*cfg.sub_num_each_cross
    val_sub = [i for i in range(
        val_start_sub, val_start_sub+cfg.sub_num_each_cross)]

    # 求 all_sub 与 val_sub 的差集，也即tr_sub
    tr_sub = [i for i in all_sub if i not in val_sub] + \
        [i for i in val_sub if i not in all_sub]
    print(' val_sub:{}'.format(val_sub))
    print(' tr_sub:{}\n'.format(tr_sub))


    # 若数据存在，则直接读取即可
    # 若要用两种数据，即cfg_2有值的时候，也即2D和3D数据一起，则这两种肯定数据肯定存在于文件夹，直接读取即可
    # 若这两种数据不存在，则需要先单独运行对应的cfg来生成一种数据
    if os.path.exists(cfg.process_dt_dir + str(cfg.sub_num-1)+ 'dt.npy'):
        print(' Data has already been processed, now, loading them from file {}......'.format(cfg.process_dt_dir))
        # 从文件读入数据
        for i in range(1,cfg.sub_num):
            dt[i] = np.load(cfg.process_dt_dir + str(i)+ 'dt.npy')
            lab[i] = np.load(cfg.process_dt_dir + str(i)+ 'lab.npy')
        # 将字典形式的数据，分为 train 和 val 两部分，并将其融合为数组形式
        tr_dt, tr_lab, val_dt, val_lab = split_tr_val(dt, lab, tr_sub, val_sub,get_tr=get_tr,get_val=get_val)
        

        # 如果cfg_2有值，则返回数据列表，以适合接下来的双输入模型
        dt_2 = {}
        if cfg_2:
            for i in range(cfg_2.sub_num):
                dt_2[i] = np.load(cfg_2.process_dt_dir + str(i)+ 'dt.npy')

            tr_dt_2,  __,    val_dt_2,  __     = split_tr_val(dt_2, lab, tr_sub, val_sub)
            tr_dt = [tr_dt,tr_dt_2]
            val_dt = [val_dt,val_dt_2]
    else:

        # 读取所有数据, 返回字典形式
        dt, lab = load_dataset(cfg)

        # 取出相应电极数据，返回字典形式
        dt = extract_elec_data(dt, cfg)

        for i in range(1,cfg.sub_num):

            # 将 每一个sub的label 转换为模型需要的二值输入格式
            lab[i] = to_categorical(np.squeeze(lab[i], axis=0))

            # 若需要，则转换为3D形式
            if cfg.dt_fm == '3D':
                dt[i] = to_3D(cfg, dt[i])

            # 为了输入模型，需要在最后一维添加通道
            dt[i] = np.expand_dims(dt[i], axis=-1)

            # 将数据存入文件
            print('{} sub data has been preprocessed, saving them to file .......'.format(i))
            np.save(cfg.process_dt_dir + str(i)+ 'dt.npy',dt[i])
            np.save(cfg.process_dt_dir + str(i)+ 'lab.npy',lab[i])


        # 分为训练集和验证集 （未打乱, 形状为：(samples，chans，points, 1) (1, samples), numpy array 形式
        tr_dt, tr_lab, val_dt, val_lab = split_tr_val(dt, lab, tr_sub, val_sub,get_tr=get_tr,get_val=get_val)
    if type(tr_dt) is list:
        print('\nAquire tr_dt: {}, tr_lab: {}, val_dt: {}, val_lab: {}\n'.format(type(tr_dt), tr_lab.shape, type(val_dt),val_lab.shape))
    else:
        print('\nAquire tr_dt: {}, tr_lab: {}, val_dt: {}, val_lab: {}\n'.format(tr_dt.shape,tr_lab.shape,val_dt.shape,val_lab.shape))

    return tr_dt, tr_lab, val_dt, val_lab


def load_dataset(cfg):
    ''' 
        因为不同的数据文件有不同的格式，可将data统一格式，
        变为：一个字典，key 为 sub，value 为 data
        如：data 变为： {0：(samples，chans，points)，1：(samples，chans，points)  ... }
        label变为：     {0：(1, samples)，            1：(1, samples)  ... }
    '''
    print('\n', 'In function load_dataset: Start load data from {} ......'.format(cfg.dataset_path))

    # 读出数据，有不同的形状
    data = None
    label = None
    if cfg.dt == 'Covert1s'or cfg.dt =='Covert2s':
        # (1, sub) (samples, points, chans) label: (1, sub)(1, samples)
        my_file = sio.loadmat(cfg.dataset_path)['TotalCell']

        # 若为我自己处理的数据
        data = my_file['Data'][0,0]   # (1, 8) (1198, 400或200, 62)
        label = my_file['Label'][0,0]  # (1, 8) (1, 1198)

        # 若为老师之前处理的数据
        # data = my_file['Data'][0][0]  # (1, 8) ( 200, 62, 1198)
        # label = my_file['Label'][0][0]  # (1, 8) (1, 1198)

    elif cfg.dt == 'SimulEEG':
        # (1, sub) (points, chans, samples) label: (1, sub)(1, samples)
        my_file = sio.loadmat(cfg.dataset_path)
        data = my_file['Data']
        label = my_file['Label']


    # 将所有的data变为 字典： 
    # {0：(samples，chans，points)，1：(samples，chans，points)  ...  8：...}
    # {0：(1, samples)，            1：(1, samples)  ...              8：...}
    sub_num = data.shape[1]
    all_dt = {}  
    all_lab = {}  
    for sub in range(sub_num):
        tmp_dt = None
        if cfg.dt == 'Covert1s'or cfg.dt =='Covert2s':
            tmp_dt = np.transpose(data[0, sub], (0, 2, 1))
        else:
            tmp_dt = np.transpose(data[0, sub], (2, 1, 0))
        tmp_lab = label[0, sub]
        all_dt[sub] = tmp_dt
        all_lab[sub] = tmp_lab

    print('     Return all_dt, all_lab: {}， {} \n'.format(
        type(all_dt), type(all_lab)))
    return all_dt, all_lab


def extract_elec_data(dt, cfg):
    ''' 从整体数据中提取特定电极数据
        Input:
            dt: 形状为：(sub, samples，chans，points)
            lab: 形状为：(sub, 1, samples)
            cfg: 主要用到 cfg.elec 
    '''
    print('\n', 'In function extract_elec_data: Start extract electrodes ...... ')

    for sub in dt.keys():
        dt[sub] = dt[sub][:, cfg.elec, :]

    print('     Return dt: {}\n'.format(type(dt)))
    return dt


def split_tr_val(dt, lab, tr_sub, val_sub,get_tr=True,get_val=True):
    '''分离 train validation 数据
    Input:
        dt: 待分裂数据，字典形式，每个 value 形状为(samples，chans，points) 
        tr_sub: 列表形式，训练 sub 的序号
        val_sub: 列表形式，验证 sub 的序号
    Output:
        tr_data
        val_data
        tr_lab
        val_lab
        以上都没有打乱，可后期增加 shuffle 参数来控制要不要打乱
    '''
    print('\n', 'In function split_tr_val: Start spliting training data and validate data ......')

    # 初始化变量
    tr_dt = np.array([])  # 未打乱的 train 数据
    tr_lab = np.array([])  # 所有未打乱的 train 标签
    val_dt = np.array([])  # 所有未打乱的 validation 数据
    val_lab = np.array([])  # 所有未打乱的 validation 标签

    # 把每个 train 被试数据拼在一起，形状为：(samples，chans，points) (1, samples)
    if get_tr:
        for i, sub in enumerate(tr_sub):
            print('     concatenate {} sub as training data...'.format(sub))
            tmp_dt = dt[sub]
            tmp_lab = lab[sub]
            if i == 0:
                tr_dt = tmp_dt
                tr_lab = tmp_lab
            else:
                tr_dt = np.concatenate((tr_dt, tmp_dt), axis=0)
                tr_lab = np.concatenate((tr_lab, tmp_lab), axis=0)

    # 并把每个 validation 被试数据拼在一起，形状为：(samples，chans，points) (1, samples)
    if get_val:
        for i, sub in enumerate(val_sub):
            print('     concatenate {} sub as validation data...'.format(sub))
            tmp_dt = dt[sub]
            tmp_lab = lab[sub]
            if i == 0:
                val_dt = tmp_dt
                val_lab = tmp_lab
            else:
                val_dt = np.concatenate((val_dt, tmp_dt), axis=0)
                val_lab = np.concatenate((val_lab, tmp_lab), axis=0)

    print('     Return tr_dt:{},  tr_lab:{},  val_dt:{},  val_lab:{}\n'.format(tr_dt.shape, tr_lab.shape, val_dt.shape, val_lab.shape))

    return tr_dt, tr_lab, val_dt, val_lab


def to_3D(cfg,tr_dt):
    print('\n', 'In function to_3D: Start converting data to 3D format ......')
    
    # 从 (samples，chans，points, 1) 压缩到 (samples，chans，points)
    tr_dt = np.squeeze(tr_dt)

    # 定义 每个电极 在空间二维矩阵中的 坐标
    SimulEEG_index_to_locate = {
        0: [0, 3],
        1: [1, 2],
        2: [1, 4],
        3: [2, 3],
        4: [3, 2],
        5: [3, 3],
        6: [4, 0],
        7: [4, 2],
        8: [4, 4],
        9: [5, 1],
        10: [5, 3],
        11: [6, 0],
        12: [6, 2],
        13: [6, 4],
        14: [7, 4],
        15: [8, 3],
        16: [0, 5],
        17: [1, 6],
        18: [2, 5],
        19: [3, 5],
        20: [3, 7],
        21: [4, 6],
        22: [4, 8],
        23: [5, 5],
        24: [5, 7],
        25: [6, 6],
        26: [6, 8],
        27: [8, 6]
    }

    Covert_index_to_locate = {
        0: [8, 4],
        2: [8, 3],
        3: [0, 3],
        4: [0, 5],
        5: [8, 5],
        7: [0, 1],
        8: [1, 1],
        9: [1, 2],
        10: [1, 3],
        11: [1, 4],
        12: [1, 5],
        13: [1, 6],
        14: [1, 7],
        15: [0, 7],
        16: [7, 1],
        17: [2, 1],
        18: [2, 2],
        19: [2, 3],
        20: [2, 4],
        21: [2, 5],
        22: [2, 6],
        23: [2, 7],
        24: [7, 7],
        25: [3, 0],
        26: [3, 1],
        27: [3, 2],
        28: [3, 3],
        29: [3,4],
        30: [3,5],
        31:[3,6],
        32:[3,7],
        33:[3,8],
        34:[4,0],
        35:[4,1],
        36:[4,2],
        37:[4,3],
        38:[4,4],
        39:[4,5],
        40:[4,6],
        41:[4,7],
        42:[4,8],
        43:[6,0],
        44:[5,0],
        45:[5,1],
        46:[5,2],
        47:[5,3],
        48:[5,4],
        49:[5,5],
        50:[5,6],
        51:[5,7],
        52:[5,8],
        53:[6,8],
        54:[7,2],
        55:[6,3],
        56:[6,4],
        57:[6,5],
        58:[7,6],
        59:[7,3],
        60:[7,4],
        61:[7,5]
    }

    def Convert_vector_to_matrix(my_dt, cfg ):
        '''将数据的 电极 填入空间二维矩阵相应坐标, 返回填充过0的三维数据表示'''
        # 初始化3维矩阵数据：全为 0
        results = np.zeros(shape=(my_dt.shape[0], 9, 9, cfg.sam_pnts))
        for sample in range(my_dt.shape[0]):
            for i in cfg.elec:
                locate = None
                # SimulEEG 数据中
                if cfg.dt == 'SimulEEG':
                    locate = SimulEEG_index_to_locate[i]
                    results[sample, locate[0], locate[1], :] = my_dt[sample, i, :]
                # Covert数据中
                elif cfg.dt == 'Covert1s' or cfg.dt == 'Covert2s':
                    locate = Covert_index_to_locate[i]
                    # 因为没有1和6电极，所以二维数据表示中会少 2行
                    # 所以要将其中的数据和 Covert_index_to_locate 中的keys对齐
                    tmp = 0
                    if i>1 and i<6:
                        tmp = i-1
                    elif i>6:
                        tmp = i-2
                    results[sample, locate[0], locate[1], :] = my_dt[sample, tmp, :]
        return results

    # 将训练数据、验证数据的 电极 填入二维矩阵相应坐标，得到填充过0的三维数据表示
    train_dt = Convert_vector_to_matrix(tr_dt,cfg)
    print('     shape after 3D transform：', end=' ')
    print(train_dt.shape)


    def Interpolate_data(my_dt,my_dt_pre,cfg):
        '''对每个样本进行插值
        Input:
            my_dt: 三维数据表示，不够的地方填充为0 的数据
            my_dt_pre: 二维数据表示
        '''
        # 初始化插值后的矩阵
        results = np.zeros(
            shape=(my_dt.shape[0], cfg.inter_size, cfg.inter_size, 400))
        for i in range(my_dt.shape[0]):
            if i % 100 == 0:
                print('             interpolating the {}th training sample, total {} '.format(
                    i+1, my_dt.shape[0]))

            #若为3D整体的插值，则对每个样本进行插值
            if cfg.interpolate_way[:2] == '3D':
                results[i] = interp3d_station(my_dt[i])
            else:
                # 对每一个 9*9 都进行2D插值，共 cfg.smp_pnts 个
                for j in range(my_dt.shape[-1]):

                    # 生成电极对应的二维坐标表示，有多少个电极，就有多少个坐标
                    x,y = None,None
                    if cfg.dt=='SimulEEG':
                        x = np.array([val[0] for val in SimulEEG_index_to_locate.values()])
                        y = np.array([val[1] for val in SimulEEG_index_to_locate.values()])
                    elif cfg.dt == 'Covert1s' or cfg.dt == 'Covert2s':
                        x = np.array([val[0] for val in Covert_index_to_locate.values()])
                        y = np.array([val[1] for val in Covert_index_to_locate.values()])

                    # 28的向量，即每个坐标的值
                    fvals = my_dt_pre[i, :, j]

                    # 生成的 9*9矩阵 （包含0）
                    fvals_990 = my_dt[i, :, :, j]

                    # 进行2D插值
                    fnew = interp2d_station_to_grid(
                        x, y, fvals,fvals_990, cfg.inter_size, cfg.interpolate_way)

                    # 对结果矩阵进行赋值
                    results[i, :, :, j] = fnew

                    '''
                    # 仅调试时使用：画出原来包含0的 9*9矩阵
                    pl.subplot(121)
                    im1 = pl.imshow(fvals_990, extent=[-1, 1, -1, 1], cmap=mpl.cm.hot,
                                    interpolation='nearest', origin="lower")  # pl.cm.jet
                    pl.colorbar(im1)
                    # 仅调试时使用：画出插值后的9*9矩阵
                    pl.subplot(122)
                    im2 = pl.imshow(
                        fnew, extent=[-1, 1, -1, 1], cmap=mpl.cm.hot, interpolation='nearest', origin="lower")
                    pl.colorbar(im2)
                    pl.show()
                    '''
        return results

    # 若要进行插值
    if cfg.is_interpolate:
        print('\n', '           Start interpolating in the 3D data ...... ', '\n')
        tr_dt_inter = Interpolate_data(train_dt,tr_dt,cfg)
        print('             shape after interpolate: {} '.format(tr_dt_inter.shape))
        # 赋值，以方便最后一起 expand_dims
        train_dt = tr_dt_inter

    print('     return data:{}'.format(train_dt.shape))

    return train_dt


# 改编自 https://blog.csdn.net/weixin_43718675/article/details/103497930
def interp2d_station_to_grid(lon, lat, data,dt_0mtr, inter_size, method='cubic'):
    '''
    func : 将站点数据插值到等经纬度格点
    inputs:
        lon: x
        lat: y
        data: 坐标（x,y） 对应的值
        fvals_990: 不足处填充了0的 inter_size*inter_size 的二维矩阵
        method: 所选插值方法，默认 0.125
    return:

        [lon_grid,lat_grid,data_grid]
    '''

    if method=='average':
        # 对这个矩阵做均值滤波，并且保留有的值（相当于均值插值）
        dt_0mtr_pad = np.zeros((dt_0mtr.shape[0]+2,dt_0mtr.shape[1]+2))
        dt_aver = np.zeros((dt_0mtr.shape[0], dt_0mtr.shape[1]))
        dt_0mtr_pad[1:-1,1:-1] = dt_0mtr
        for i in range(1,dt_0mtr_pad.shape[0]-1):
            for j in range(1,dt_0mtr_pad.shape[1]-1):
                dt_aver[i-1,j-1] = np.mean(dt_0mtr_pad[i-1:i+2,j-1:j+2])
        # 把 dt_0mtr 中的不为0的元素赋值给 dt_aver,即保留原来的值
        for i in range(dt_0mtr.shape[0]):
            for j in range(dt_0mtr.shape[1]):
                if dt_0mtr[i,j]!=0:
                    dt_aver[i,j] = dt_0mtr[i,j]
        return dt_aver

    # step1: 先将 lon,lat,data转换成 n*1 的array数组
    lon = np.array(lon).reshape(-1, 1)
    lat = np.array(lat).reshape(-1, 1)

    # 转换为 (n,2) 的数组，也即散点坐标
    points = np.concatenate([lon, lat], axis=1)

    # 散点坐标对应的值
    data = np.array(data).reshape(-1, 1)

    # 定义插值之后的矩阵网格大小
    lon_grid, lat_grid = np.mgrid[0:inter_size, 0:inter_size]

    # 进行网格插值,得到 lon_grid,lat_grid 大小的矩阵
    grid_data = griddata(points, data, (lon_grid, lat_grid), method=method)
    grid_data = grid_data[:, :, 0]

    return grid_data


def interp3d_station(dt):
    '''
    :param dt: 一个样本值，（9,9,400），不够的地方填充0
    :return: 经过插值后的样本 （9,9,400），目前只实现了平均插值方法
    '''
    # 在样本四周添加0
    dt_pad = np.zeros(shape=(dt.shape[0]+2,dt.shape[1]+2,dt.shape[2]+2))
    dt_pad[1:-1,1:-1,1:-1] = dt
    # 进行均值滤波
    results = np.zeros(shape=dt.shape)
    for i in range(1,dt_pad.shape[0]-1):
        for j in range(1,dt_pad.shape[1]-1):
            for k in range(1,dt_pad.shape[2]-1):
                results[i-1,j-1,k-1] = np.mean(dt_pad[i-1:i+2,j-1:j+2,k-1:k+2])
    # 把 dt_0mtr 中的不为0的元素赋值给 dt_aver,即保留原来的值
    for i in range(dt.shape[0]):
        for j in range(dt.shape[1]):
            for k in range(dt.shape[2]):
                if dt[i, j,k] != 0:
                    results[i, j,k] = dt[i, j,k]
    return results

