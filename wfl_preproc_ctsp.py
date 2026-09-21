# -*- coding: utf-8 -*-
"""
CTSP (Common Temporal Subspace Projection) —— 按论文重写的版本

论文
    T. Watanabe, Y. Kawabata, D. Ukegawa, S. Kawabata, Y. Adachi, K. Sekihara,
    "Removal of Stimulus-Induced Artifacts in Functional Spinal Cord Imaging",
    35th Annual International Conference of the IEEE EMBS, Osaka, 2013,
    pp. 3391-3394.                       (本目录下的 ctsp.pdf)

论文方法 (II. METHOD, 式(1)-(8))
    测量模型:  B = B_S + B_I                    (1)   M 个传感器 x T 个时间点
               A = B_I + B_A                    (4)   artifact(control) 测量

    对 B 做 SVD:  B = [d_1..d_M] diag(l_1..l_M) [e_1..e_M]^T        (5)
    对 A 做 SVD:  A = [f_1..f_M] diag(r_1..r_M) [g_1..g_M]^T        (3)
    其中 d, f 是"空间"奇异向量, e, g 是"时间"奇异向量。

    论文明确取时间子空间("we are interested in the subspace spanned by temporal
    vectors"):
        E_q = [e_1, ..., e_q]   (T x q)        (6)   q = B 的显著大奇异值个数
        G_p = [g_1, ..., g_p]   (T x p)        (7)   p = A 的显著大奇异值个数
    干扰子空间 = 两个时间子空间的交集 S_A ∩ S_B, 由式(8)求:
        Y^T (E_q^T G_p) Z = diag(cos(theta_1), cos(theta_2), ...)
    该对角元 = 两个子空间的主角余弦; cos(theta) ≈ 1 的个数即交集维数 r。

    交集的正交基 (T x r):
        U_r = [E_q Y] 的前 r 列            (也可以取 [G_p Z] 的前 r 列)
    干扰抑制: 在"时间"方向右乘投影算子
        B_clean = B (I - U_r U_r^T)


与原文件 wfl_preproc_ctsp.py 的差异(原文件未改动, 本文件是副本)
    1) 【关键】原文件 flag==1 分支把数据转置后做 SVD, 取的是右奇异向量 V, 也就是
       "空间"奇异向量; 它算的是两次测量"空间子空间"的交集, 投影也发生在空间方向,
       而论文用的是"时间子空间"的交集(式(6)-(8))以及右乘的 B(I - U_r U_r^T)。
       两者不是同一个方法。本文件按论文改用时间奇异向量。
       旁证: 原文件里 "C = Q_in.T @ Q_out" 上方的注释写的是 common temporal
       subspace, 但 Q_in / Q_out 实际是空间向量, 其打印信息也写着 "spatial
       dimensions" —— 移植时空间/时间弄反了。
    2) 原文件 flag==0 分支(移植自 MNE 的 tSSS 空房间投影)在数学上确实是"时间子空间
       的交集", 与论文是同一思路(只是用 QR 正交基 + 相关系数阈值代替截断 SVD);
       但该分支在原文件里根本跑不通:
         a. 变量 B 只在 flag==1 里定义(原文件是 B = data_signal_all[...].T), flag==0
            里直接使用 B, 会抛 UnboundLocalError: cannot access local variable 'B';
         b. 即使把 B 补成 data_signal_all[picks_good, :], 最后写回 raw 时还有一句
            raw_clean._data[picks_good, :] = Bclean.T, 多了一个 .T, 当
            n_times != n_channels 时会报 shape 错(两者相等时会静默出错)。
       本文件已修正, 并保留为 st_correlation 分支。
    3) 论文 III.B 节在 CSP 之前还有一步 per-channel 的 power correction(把 artifact
       测量的功率按通道最小二乘对齐到 SCEF 测量), 原文件缺失, 本文件已加入
       (power_correction=True, 可关闭)。注意: 逐通道缩放不改变 A 的行空间(= 时间
       子空间), 因此它影响的是"哪些分量排在前 q/p 位"以及奇异值谱的形状, 从而会改变
       q/p/r 的判定; 若你沿用原文件的做法(不对 A 做功率校正), 可设 False。
    4) 参数检查: 原文件在 Nout/Nin/Nee 只给出一部分且 st_correlation=None 时会让
       flag 未定义(UnboundLocalError); 本文件给出明确的 ValueError。
    5) 新增: 两个 raw 的通道顺序检查与自动对齐; 诊断量输出(return_diag=True)。

用法(与原文件一致, 可直接替换):
    raw_clean = ctsp(raw, raw_room, picks=None, Nin=q, Nout=p, Nee=r)
    raw_clean, diag = ctsp(raw, raw_room, Nin=2, Nout=2, Nee=0.999,
                           return_diag=True)   # 用 cos_theta 帮助确定 r
"""

import warnings

import numpy as np

from mne.fixes import _safe_svd
from mne.io.pick import _picks_to_idx


# --------------------------------------------------------------------------- #
# 交互式选点(可选, 需要图形界面)
# --------------------------------------------------------------------------- #
click_x = None
click_y = None


def onclick(event):
    global click_x, click_y
    if event.button == 1:  # 鼠标左键
        click_x = event.xdata
        click_y = event.ydata


def get_click_position(y, linestyle='-', marker='o'):
    # 延迟导入: 无 GUI/无 matplotlib 的环境下也能 import 本模块
    import matplotlib.pyplot as plt

    global click_x, click_y
    fig, ax = plt.subplots()
    ax.plot(y, linestyle=linestyle, marker=marker)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    while click_x is None or click_y is None:
        plt.pause(0.1)
    fig.canvas.mpl_disconnect(cid)
    x = click_x
    click_x = None
    click_y = None
    plt.close(fig)
    return round(x)


def _orth_overwrite(A):
    """Create a slightly more efficient 'orth'."""
    # adapted from scipy/linalg/decomp_svd.py
    check_disable = dict(check_finite=False)
    u, s = _safe_svd(A, full_matrices=False, **check_disable)[:2]
    M, N = A.shape
    eps = np.finfo(float).eps
    tol = max(M, N) * np.amax(s) * eps
    num = np.sum(s > tol, dtype=int)
    return u[:, :num]


def _power_correction(B, A):
    """论文 III.B 节的 power correction(逐通道最小二乘幅度匹配)。

        w_i = argmin_w sum_j (B_ij - w * A_ij)^2 = <B_i, A_i> / <A_i, A_i>

    返回按通道缩放后的 artifact 测量以及权重 w。
    """
    num = np.sum(B * A, axis=1)
    den = np.sum(A * A, axis=1)
    w = np.ones_like(den)
    nz = den > 0
    w[nz] = num[nz] / den[nz]
    return A * w[:, None], w


def _align_channels(raw, raw_room):
    """保证 artifact 测量与 SCEF 测量的通道顺序一致。"""
    if list(raw_room.ch_names) == list(raw.ch_names):
        return raw_room
    if set(raw_room.ch_names) != set(raw.ch_names):
        raise ValueError('raw and raw_room must contain the same channels')
    return raw_room.copy().reorder_channels(raw.ch_names)


def _select_ncomp(N, s, name):
    """分量个数 q, p 的确定方式(与原文件保持一致)。"""
    if N is None:
        N = int(input('enter %s: ' % name))
    elif isinstance(N, str) and N == 'interactive':
        N = get_click_position(np.log10(s))
    elif isinstance(N, str) and N == 'all':
        eps = np.finfo(float).eps
        N = int(np.sum(s / s[0] > 1e5 * eps))
    N = int(N)
    if N < 1:
        raise ValueError('%s must be >= 1, got %s' % (name, N))
    return N


def _select_r(Nee, cos_theta, st_correlation):
    """公共(干扰)子空间的维数 r: 论文取 cos(theta) ≈ 1 的个数。"""
    if Nee is None:
        Nee = st_correlation          # 沿用原文件的阈值语义
    if isinstance(Nee, str):
        if Nee == 'auto':
            raise ValueError('automatic determination of intersection dimension '
                             'is not supported')
        if Nee == 'interactive':
            Nee = get_click_position(np.log10(cos_theta))
        else:
            Nee = float(Nee)
    if Nee < 1:                       # 数值 < 1 视为 cos(theta) 阈值
        r = int(np.sum(cos_theta >= Nee))
    else:
        r = int(Nee)
    return int(min(r, len(cos_theta)))


def ctsp(raw, raw_room, picks=None, Nout=None, Nin=None, Nee=None,
         st_correlation=0.98, power_correction=True, return_diag=False):
    """Common Temporal Subspace Projection (Watanabe et al., EMBC 2013).

    Parameters
    ----------
    raw : mne.io.Raw
        SCEF 测量 B = B_S + B_I。
    raw_room : mne.io.Raw
        artifact(control) 测量 A = B_I + B_A, 通道与 raw 一致。
        注意: 论文求的是"时间子空间"的交集, 所以 raw_room 必须与 raw 有相同的
        采样率、相同的采样点数、并且时间对齐(例如都取刺激后同一个时间窗);
        长度不同时请先把两者 crop/重采样到同一个窗口。
    picks : str | list | None
        参与处理的通道, 默认全部(坏道保留原值, 不参与投影)。
    Nin : int | 'all' | 'interactive' | None
        式(6)中 q: SCEF 测量用于张成时间子空间的时间奇异向量个数。
    Nout : int | 'all' | 'interactive' | None
        式(7)中 p: artifact 测量用于张成时间子空间的时间奇异向量个数。
    Nee : int | float | 'interactive' | None
        交集维数 r。整数 >= 1 时直接使用; 数值 < 1 时作为 cos(theta) 的阈值;
        None 时使用 st_correlation 作为阈值(论文用的是 cos(theta) ≈ 1, 因此
        想更贴近论文可以取 0.999 之类)。
    st_correlation : float | None
        cos(theta) 阈值(默认 0.98)。当 Nout, Nin, Nee 都给出时, 该参数只用于
        确定 r(见 Nee)。为 None 时只能走 Nout/Nin/Nee 分支。
    power_correction : bool
        是否先做论文 III.B 节的逐通道功率校正(默认 True, 与论文一致)。
    return_diag : bool
        为 True 时额外返回诊断字典(含 cos_theta, q, p, r, U_r 等)。

    Returns
    -------
    raw_clean : mne.io.Raw
    diag : dict (仅当 return_diag=True)
    """
    # ------------------------------------------------------------------ #
    # 分支判定(与原文件兼容: 三个维数都给出 -> 论文式(6)-(8)的截断 SVD 分支)
    # ------------------------------------------------------------------ #
    if Nout is not None and Nin is not None and Nee is not None:
        flag = 1
    elif st_correlation is None:
        raise ValueError('Need to specify st_correlation or all of '
                         'Nout, Nin, Nee')
    else:
        if st_correlation < 0 or st_correlation > 1:
            raise ValueError('st_correlation must be between 0 and 1')
        if not (Nout is None and Nin is None and Nee is None):
            warnings.warn('Nout/Nin/Nee are only used when all three of them '
                          'are given; falling back to the st_correlation '
                          '(QR) branch, as in the original file.')
        flag = 0

    picks = _picks_to_idx(raw.info, picks, exclude=())
    picks_good, picks_bad = list(), list()  # these are indices into picks
    for ii, pi in enumerate(picks):
        if raw.ch_names[pi] in raw.info["bads"]:
            picks_bad.append(ii)
        else:
            picks_good.append(ii)
    picks_good = np.array(picks_good, int)
    picks_bad = np.array(picks_bad, int)

    raw_room = _align_channels(raw, raw_room)

    # B, A: M x T (传感器 x 时间), 与论文的定义一致
    B = raw.get_data()[picks_good, :]
    A = raw_room.get_data()[picks_good, :]

    # 时间子空间的交集要求两个测量共用同一条时间轴(采样率/点数/对齐都要一致)
    if abs(float(raw.info['sfreq']) - float(raw_room.info['sfreq'])) > 1e-6:
        raise ValueError('raw and raw_room must have the same sampling '
                         'frequency (got %g and %g)'
                         % (float(raw.info['sfreq']),
                            float(raw_room.info['sfreq'])))
    if B.shape[1] != A.shape[1]:
        raise ValueError(
            'raw and raw_room must have the same number of time points '
            '(got %d and %d). CTSP intersects the TEMPORAL subspaces of the '
            'two measurements, which are defined on a shared time axis, so '
            'crop (or resample) both to the same stimulus-aligned window '
            'before calling ctsp.'
            % (B.shape[1], A.shape[1]))

    diag = dict(picks_good=picks_good, picks_bad=picks_bad)
    if power_correction:
        A, w = _power_correction(B, A)
        diag['power_w'] = w

    if flag == 1:
        # ---------------- 论文 II.B: 时间子空间的交集 ---------------- #
        # B = U_B diag(s_B) Vt_B, A = U_A diag(s_A) Vt_A
        # 时间奇异向量 = Vt 的行 -> E = Vt_B.T (T x K) 即论文式(5)的 e_1..e_M
        _, s_B, Vt_B = _safe_svd(B, full_matrices=False)
        _, s_A, Vt_A = _safe_svd(A, full_matrices=False)
        E = Vt_B.T                      # (T x min(M,T)) 时间基
        G = Vt_A.T                      # (T x min(M,T)) 时间基

        q = _select_ncomp(Nin, s_B, 'the number of temporal dimensions '
                                     'for the SCEF data, q')
        p = _select_ncomp(Nout, s_A, 'the number of temporal dimensions '
                                     'for the artifact data, p')
        print('Using q = %d (SCEF) and p = %d (artifact) temporal dimensions'
              % (q, p))

        if q >= E.shape[1] or p >= G.shape[1]:
            warnings.warn('q or p reaches the dimension of the temporal basis '
                          '(%d); the intersection may swallow the whole data. '
                          'Choose q/p from the "distinctively large" singular '
                          'values as in the paper.' % E.shape[1])

        E_q = E[:, :q]                  # 式(6)
        G_p = G[:, :p]                  # 式(7)

        # 式(8): 主角余弦
        Y, cos_theta, Zt = _safe_svd(E_q.T @ G_p, full_matrices=False)
        Z = Zt.T

        r = _select_r(Nee, cos_theta, st_correlation)
        print('Using r = %d common temporal dimensions' % r)
        if r == 0:
            warnings.warn('No common temporal component above the threshold; '
                          'the data are returned unchanged. Try a smaller '
                          'threshold or larger q/p.')

        U_r = E_q @ Y[:, :r]            # 论文: U_r = [E_q Y] 的前 r 列 (T x r)
        # 论文指出也可取 [G_p Z] 的前 r 列, 二者张成同一个子空间:
        # U_r_alt = G_p @ Z[:, :r]

        B_clean = B - (B @ U_r) @ U_r.T         # B (I - U_r U_r^T)

        diag.update(dict(s_B=s_B, s_A=s_A, E=E, G=G, q=q, p=p, r=r,
                         cos_theta=cos_theta, Y=Y, Z=Z, U_r=U_r))
    else:
        # ------- 备选分支: MNE tSSS 风格(时间子空间交集 + 相关系数阈值) ------- #
        from scipy import linalg
        check_disable = dict(check_finite=False)

        if A.shape[1] <= A.shape[0]:
            warnings.warn('The number of time points (%d) is not larger than '
                          'the number of channels (%d); the temporal subspaces '
                          'may coincide and remove everything.'
                          % (A.shape[1], A.shape[0]))

        data_int = B
        data_res = A

        n = np.linalg.norm(data_int)
        n = 1.0 if n == 0 else n  # all-zero data should gracefully continue
        data_int = _orth_overwrite((data_int / n).T)
        n = np.linalg.norm(data_res)
        n = 1.0 if n == 0 else n
        data_res = _orth_overwrite((data_res / n).T)

        Q_int = linalg.qr(data_int, overwrite_a=True, mode="economic",
                          **check_disable)[0].T
        Q_res = linalg.qr(data_res, overwrite_a=True, mode="economic",
                          **check_disable)[0]
        C_mat = np.dot(Q_int, Q_res)
        del Q_int
        cos_theta, Vh_intersect = _safe_svd(C_mat, full_matrices=False,
                                            **check_disable)[1:]
        del C_mat
        intersect_mask = cos_theta >= st_correlation
        del cos_theta
        Vh_intersect = Vh_intersect[intersect_mask].T
        U_r = np.dot(Q_res, Vh_intersect)        # (T x r) 时间基
        r = U_r.shape[1]
        print('Using r = %d common temporal dimensions' % r)

        B_clean = B - (B @ U_r) @ U_r.T          # B (I - U_r U_r^T)

        diag.update(dict(r=r, U_r=U_r))

    raw_clean = raw.copy()
    raw_clean._data[picks_good, :] = B_clean     # B_clean 是 M x T, 与 _data 同向

    if return_diag:
        return raw_clean, diag
    return raw_clean
