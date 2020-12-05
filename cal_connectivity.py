'''from preprocess_data import load_data
from config import config

from mne.connectivity import spectral_connectivity
from mne.viz import plot_connectivity_circle
import numpy as np
# 我们一般的数据格式，N,elecs,points,chans
cfg = config(
    # 从 datasets 中选择数据集 datasets = ['Covert1s','Covert2s', 'SimulEEG']
    dt_idx = 2,
    # 从 Covert_elec 和 SimulEEG_elec 中选择电极
    elec_area = 'All',
    # 设定数据的输入方式： '2D' or '3D' ( 只有当 elec_area 为 All时，才能为 3D )
    dt_fm = '2D',
    is_Z_Norm = False,
    cal_con=False,
    # 从 model 文件中选择模型   
    mdl_nm = 'EegnetOrigin',
    # 每个模型跑几次, 只关心几次，不要关心是从0还是从1开始
    cycles = 10,
    # 开始的cycle 和 开始的cross，此处的概念都是从1开始
    # 注意，若之前已经训练了模型，那么只能接着训练，记得调此处
    # 不然，像 混淆矩阵这些文件会堆积，会出错
    st_cycle = 4,
    st_cross = 1,
    initAlpha=0.001,
    # 自己写创建的文件夹备注信息，3D模型是否插值，插值方法等，都要在这里表明
    other_info = '进行网络图实验，可删'
)

cfg.set_dir(1,1)

# load 数据，第一个被试作为 val，之后的画图都只基于 val
tr_x, tr_y, val_x, val_y = load_data(1, cfg) 

# 得到 att 和 non_att 的数据
# 对于SimulEEG，att label 为 0，non-att label 为 1
# 对于covert，att label 为 1，non-att label 为 0
y = np.where(val_y==1)[1]
idx_0 = np.where(y==0)
idx_1 = np.where(y==1)
att_dt = val_x[idx_0]
non_att_dt = val_x[idx_1]


# 在所有注意力数据上生成一个网络
con, freqs, times, n_epochs, n_tapers = spectral_connectivity(att_dt[:,:,:,0], \
                                                            method='pli', mode='multitaper',faverage=True)

# 利用圆形图可视化连接权重
plot_connectivity_circle(np.squeeze(con),range(1,29))


# 在所有非注意力数据上生成一个网络
con, freqs, times, n_epochs, n_tapers = spectral_connectivity(non_att_dt[:,:,:,0], \
                                                            method='pli', mode='multitaper',faverage=True)
# 利用圆形图可视化连接权重
plot_connectivity_circle(np.squeeze(con),range(1,29))
'''


from scipy import signal
import numpy as np
from mne_features_master.mne_features.utils import (_idxiter, power_spectrum, _embed, _get_feature_funcs,
                    _get_feature_func_names, _psd_params_checker)


def compute_phase_lock_val(data, include_diag=False):
    """Phase Locking Value (PLV).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    include_diag : bool (default: False)
        If False, features corresponding to pairs of identical electrodes
        are not computed. In other words, features are not computed from pairs
        of electrodes of the form ``(ch[i], ch[i])``.

    Returns
    -------
    output : ndarray, shape (n_output,)
        With ``n_output = n_channels * (n_channels + 1) / 2`` if
        ``include_diag`` is True and
        ``n_output = n_channels * (n_channels - 1) / 2`` if
        ``include_diag`` is False.

    Notes
    -----
    Alias of the feature function: **phase_lock_val**. See [1]_.

    References
    ----------
    .. [1] http://www.gatsby.ucl.ac.uk/~vincenta/kaggle/report.pdf
    """
    n_channels, n_times = data.shape
    if include_diag:
        n_coefs = n_channels * (n_channels + 1) // 2
    else:
        n_coefs = n_channels * (n_channels - 1) // 2
    plv = np.empty((n_coefs,))
    for s, i, j in _idxiter(n_channels, include_diag=include_diag):
        if i == j:
            plv[j] = 1
        else:
            xa = signal.hilbert(data[i, :])
            ya = signal.hilbert(data[j, :])
            phi_x = np.angle(xa)
            phi_y = np.angle(ya)
            plv[s] = np.absolute(np.mean(np.exp(1j * (phi_x - phi_y))))
    return plv
