# -*- coding: utf-8 -*-
"""
DSSP (dual signal subspace projection) —— 按论文复现, 适配 mne.io.Raw

论文
    K. Sekihara, Y. Kawabata, S. Ushio, S. Sumiya, S. Kawabata, Y. Adachi,
    S. S. Nagarajan, "Dual signal subspace projection (DSSP): a novel algorithm
    for removing large interference in biomagnetic measurements",
    J. Neural Eng. 13 (2016) 036007.        (本目录下的 dssp.pdf)

与 ctsp / s3p 不同, DSSP 不需要单独的噪声段或伪迹段测量。它利用"信号子空间"在
空间域和时间域的对偶定义: 先从空间域伪信号子空间的里、外两侧各造一个数据矩阵,
再取这两个矩阵"行空间"(即时间域子空间)的交集作为干扰子空间, 最后把原始数据
投影到与该干扰子空间正交的方向上。

算法 (论文式(1)-(40))
    数据模型: B (M 传感器 x K 时间点) = B_S + B_I + B_e, 假设 M < K   (式(5))

    1) 空间域伪信号子空间 (2.2 节, 式(8)-(12))
       把源空间离散成 N 个体素; 体素 r_n 处的导向场矩阵 L(r_n) 为 M x 3
       (x/y/z 三个方向), 拼成增广导向场矩阵
           F = [L(r_1), ..., L(r_N)]           (M x 3N)                (8)
       对 Gram 矩阵做特征分解
           F F^T = E diag(gamma_1, ..., gamma_M) E^T                   (10)
       取前 z 个"明显大"的特征值对应的特征向量
           P = E_z E_z^T                                               (11)
       于是 P b_s(t) = b_s(t) (式(12)); 而落在源空间之外的干扰源只会被这个
       投影算子"钝钝地"切掉(附录 A.2 与图 20), 所以 P b_I 与 (I-P) b_I 都不为
       零 —— 这正是 DSSP 能同时从里外两侧看到干扰的前提。

    2) 里外两侧的数据矩阵 (式(20)(21))
           B_in = P B ,      B_out = (I - P) B
       它们的行空间分别是 S_S ⊕ S_I ⊕ S_e^in (式(26)) 和 S_I ⊕ S_e^out
       (式(29)); 由于 P B_e 的行与 (I-P) B_e 的行正交, 两者交集就是时间域
       干扰子空间 (式(30)):   S_I = S_in ∩ S_out

    3) 取它们的行空间 (式(31)-(34))
           B_in  = [f...] diag(xi)  [u...]^T ,  S_in  ~ span{u_1, ..., u_m}
           B_out = [g...] diag(phi) [v...]^T ,  S_out ~ span{v_1, ..., v_n}
       u_i, v_j 是 K 维(时间方向)的右奇异向量; m, n 取奇异值谱里"明显大"的个数。

    4) 求交集的基 (式(35)-(39))
           U = [u_1..u_m] (K x m),   V = [v_1..v_n] (K x n)
           U^T V = Y diag(cos(theta_1), ...) Z^T                       (37)
       奇异值就是两个行空间之间的主角余弦; 交集维数 r 由 cos(theta) ≈ 1 决定。
       取 G = [y_1..y_r] = (U Y) 的前 r 列, 它是干扰子空间的正交基 (式(38)(39))。

    5) 去掉干扰 (式(40)) —— 右乘, 作用在时间方向
           B_clean = B (I - G G^T)

参数与实测取值 (论文第 5、6 节与附录, 写在这里备忘)
    * m, n (Nin, Nout): 论文明确说这两个参数不关键, 建议"宁可取大", 因为
      span{u_1..u_m} 与 span{v_1..v_n} 里多出来的那些维数(噪声子空间成分)在求
      交集时会被自动排除(式(43)(44)那段讨论)。论文所有实测一律取 20, 模拟里
      改成 40 结果也几乎一样。本文件默认 Nin = Nout = 20。
    * r (Nee): 这是关键参数, 论文用 cos(theta) 阈值 0.99 确定: 计算机模拟
      r = 6, VNS 病人的 MEG r = 1, 真实 SCEF 数据 r = 3。r 取大一般问题不大
      (模拟里 r = 12 与 r = 6 结果接近), 只有当信号与干扰的时间过程高度相关
      (两个子空间很接近)时, 多出来的维数会顺带削掉一部分信号。
    * z (Nspace): 论文只说取"明显大"的特征值, 没有定量准则(实际用时看
      log10(gamma) 曲线的拐点)。重要的前提是: **干扰源要位于源空间之外**。
      若干扰源在源空间内(或恰好在边界上), DSSP 要么削不掉干扰, 要么在削掉
      干扰的同时牺牲大量信号强度(论文 3.2 节末尾与图 9)。
    * 优点: 不需要单独的伪迹/空房间测量(相对 CSP/CTSP 测量时间减半), 也不需要
      知道干扰源的位置(相对 SSP)。代价是必须有一个覆盖感兴趣脑区的源空间
      导向场 F。

用法
    from wfl_preproc_dssp_paper import dssp

    # leadfield: (n_channels, D) 的矩阵, 或 mne.Forward(取 gain matrix 作 F)
    raw_clean = dssp(raw, leadfield, Nspace=30)          # Nin=Nout=20, r 自动
    raw_clean = dssp(raw, fwd, Nspace=30, Nee=3)         # 指定 r = 3
    raw_clean, diag = dssp(raw, fwd, Nspace=30, return_diag=True)

    # diag: gamma(Gram 特征值), s_in/s_out(奇异值谱), cos_theta(主角余弦),
    #       z/m/n/r, P, G, picks_good ...
    # 建议先用 return_diag=True 看一眼 gamma 与 cos_theta 曲线, 再定 Nspace/Nee。

注意
    * 输入输出都是 mne.io.Raw; 坏道保留原值, 不参与投影。
    * 不需要两段等长数据(与 ctsp 不同), DSSP 只用一段数据 B。
    * 论文里的 B 是"一次测量"的数据矩阵: 模拟与实测都取刺激后的一段/一个
      epoch 窗(K = 时间点数), 并假设 M < K。
"""

import warnings

import numpy as np

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


def get_click_position(y, title='', linestyle='-', marker='o'):
    # 延迟导入: 无 GUI / 无 matplotlib 的环境下也能 import 本模块
    import matplotlib.pyplot as plt

    global click_x, click_y
    fig, ax = plt.subplots()
    ax.plot(y, linestyle=linestyle, marker=marker)
    if title:
        ax.set_title(title)
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


# --------------------------------------------------------------------------- #
# 参数处理
# --------------------------------------------------------------------------- #
def _get_leadfield(leadfield, raw):
    """把 leadfield 统一成 (n_channels, D) 的矩阵, 行顺序与 raw.ch_names 一致。

    接受 numpy 矩阵(即论文的 F = [L(r_1), ..., L(r_N)]), 也接受 mne.Forward
    (此时取它的 gain matrix; D = 3 x 体素数, 与式(8) 的 F 等价)。
    forward 里没有的通道(如 Trigger)对应的行填 NaN, 这样一旦这些通道被选进
    参与处理的通道集里, 就会得到明确的报错而不是静默出错。
    """
    sol, info = None, None
    if isinstance(leadfield, dict) and 'sol' in leadfield:
        sol, info = leadfield['sol'], leadfield.get('info')
    elif hasattr(leadfield, 'sol'):
        sol, info = leadfield.sol, getattr(leadfield, 'info', None)
    if isinstance(sol, dict) and 'data' in sol:
        gain = np.asarray(sol['data'], float)          # (n_ch_fwd, D)
        names = list(info['ch_names'])
        F = np.full((len(raw.ch_names), gain.shape[1]), np.nan)
        for i, ch in enumerate(raw.ch_names):
            if ch in names:
                F[i] = gain[names.index(ch)]
        return F
    F = np.asarray(leadfield, float)
    if F.ndim != 2:
        raise ValueError('leadfield must be a 2D array (n_channels, D)')
    return F


def _select_nspace(Nspace, gamma, space_tol):
    """伪信号子空间的维数 z (Gram 矩阵大特征值的个数)。

    None -> 命令行输入; 'interactive' -> 在 log10(gamma) 曲线上点一下;
    'auto' -> 取 gamma_i >= space_tol * gamma_1 的个数(粗略准则, 论文没给准则);
    'all' -> 全部(会退化成 P = I, 不推荐); 其余按整数处理。
    """
    if Nspace is None:
        Nspace = int(input('enter the dimension of the pseudo signal subspace '
                           '(z): '))
    elif isinstance(Nspace, str) and Nspace == 'interactive':
        Nspace = get_click_position(np.log10(gamma), title='z (Gram eigenvalues)')
    elif isinstance(Nspace, str) and Nspace == 'auto':
        Nspace = int(np.sum(gamma >= space_tol * gamma[0]))
    elif isinstance(Nspace, str) and Nspace == 'all':
        eps = np.finfo(float).eps
        Nspace = int(np.sum(gamma / gamma[0] > 1e5 * eps))
    Nspace = int(Nspace)
    if Nspace < 1:
        raise ValueError('Nspace must be >= 1, got %s' % (Nspace,))
    return Nspace


def _select_ncomp(N, s, name):
    """里外两侧行空间维数 m, n 的确定(论文建议宁可取大, 默认 20)。"""
    if N is None:
        N = int(input('enter %s: ' % name))
    elif isinstance(N, str) and N == 'interactive':
        N = get_click_position(np.log10(s), title=name)
    elif isinstance(N, str) and N == 'all':
        eps = np.finfo(float).eps
        N = int(np.sum(s / s[0] > 1e5 * eps))
    N = int(N)
    if N < 1:
        raise ValueError('%s must be >= 1, got %s' % (name, N))
    return N


def _select_r(Nee, cos_theta, st_correlation):
    """干扰子空间维数 r: 论文用 cos(theta) ≈ 1(阈值 0.99) 的个数。"""
    if Nee is None:
        return int(np.sum(cos_theta >= st_correlation))
    if isinstance(Nee, str):
        if Nee == 'interactive':
            Nee = get_click_position(np.log10(cos_theta), title='r (cos theta)')
        elif Nee == 'auto':
            return int(np.sum(cos_theta >= st_correlation))
        else:
            raise ValueError('unknown Nee: %r' % (Nee,))
    if Nee < 1:                     # 数值 < 1 视为 cos(theta) 阈值
        return int(np.sum(cos_theta >= Nee))
    return int(min(Nee, len(cos_theta)))


def _effrank(s, rank_tol):
    """奇异值谱的有效秩: 只保留相对最大奇异值不小于 rank_tol 的那些方向。

    不够这个量级的方向携带的能量可以忽略(相对功率 < rank_tol^2), 但它们的
    奇异向量在退化情形下是"零空间里的任意向量", 一旦取进行空间就会造出
    假的公共成分(假交集), 所以这里必须把它们排除掉。
    """
    if s.size == 0:
        return 0
    return int(np.sum(s > rank_tol * s[0]))


# --------------------------------------------------------------------------- #
# 主函数
# --------------------------------------------------------------------------- #
def dssp(raw, leadfield, picks=None, Nspace=None, Nin=20, Nout=20, Nee=None,
         st_correlation=0.99, space_tol=1e-3, rank_tol=1e-8,
         return_diag=False):
    """DSSP 干扰去除 (Sekihara et al., J. Neural Eng. 2016), 输入输出都是 Raw。

    Parameters
    ----------
    raw : mne.io.Raw
        待处理的数据(通常取刺激后的一段/一个 epoch 窗, 且通道数 < 时间点数)。
    leadfield : ndarray | mne.Forward
        增广导向场矩阵 F (n_channels x D, 式(8)): 覆盖源空间的导向场, D 通常是
        3 x 体素个数。也可以直接给 mne.Forward, 此时取它的 gain matrix。
        前提: 干扰源要位于这个源空间之外。
        行数可以是 raw 的全部通道数, 也可以正好是参与处理的通道数(此时约定
        行顺序与 picks 一致), 两者都行。
    picks : str | list | None
        参与处理的通道; 坏道保持原样。
    Nspace : int | 'auto' | 'interactive' | 'all' | None
        伪信号子空间维数 z(式(11))。None 时命令行输入; 'interactive' 在
        log10(特征值) 曲线上点选; 'auto' 用相对阈值 space_tol 自动判断。
    Nin, Nout : int | 'all' | 'interactive' | None
        B_in / B_out 行空间的维数 m, n(式(33)(34))。论文建议宁可取大, 实测一律
        取 20(默认)。
    Nee : int | float | 'interactive' | 'auto' | None
        干扰子空间维数 r(式(38))。None / 'auto' 时取 cos(theta) >= st_correlation
        的个数; 整数 >= 1 直接使用; 数值 < 1 视为 cos(theta) 阈值。
    st_correlation : float
        cos(theta) 阈值, 论文取 0.99。
    space_tol : float
        Nspace='auto' 时的相对特征值阈值。
    rank_tol : float
        B_in / B_out 行空间的有效秩判据(相对最大奇异值, 默认 1e-8)。m, n 超过
        有效秩时会自动收缩, 因为超出的那部分奇异向量是零空间里的任意向量,
        会制造出假的公共成分。
    return_diag : bool
        为 True 时额外返回诊断字典。

    Returns
    -------
    raw_clean : mne.io.Raw
    diag : dict (仅当 return_diag=True)
        gamma, z, s_in, s_out, m, n, cos_theta, r, P, G, U, V, picks_good 等,
        便于检查参数选得是否合适。
    """
    # ----------------------------- 数据准备 ----------------------------- #
    picks = _picks_to_idx(raw.info, picks, exclude=())
    picks_good, picks_bad = list(), list()  # these are indices into picks
    for ii, pi in enumerate(picks):
        if raw.ch_names[pi] in raw.info["bads"]:
            picks_bad.append(ii)
        else:
            picks_good.append(ii)
    picks_good = np.array(picks_good, int)
    picks_bad = np.array(picks_bad, int)
    if picks_good.size == 0:
        raise ValueError('no good channel to process')

    # B: M x K (传感器 x 时间点), 与论文一致 (式(2)(5))
    B = np.asarray(raw.get_data()[picks_good, :], float)
    M, K = B.shape
    if M >= K:
        warnings.warn('DSSP 论文假设通道数 M < 时间点数 K, 现在是 M = %d, '
                      'K = %d' % (M, K))

    F = _get_leadfield(leadfield, raw)
    if F.shape[0] == raw.info['nchan']:
        bad = [raw.ch_names[pi] for pi in picks[picks_good]
               if not np.isfinite(F[pi, :]).all()]
        F = np.asarray(F[picks_good, :], float)        # (M x D)
    elif F.shape[0] == picks_good.size:
        bad = [] if np.isfinite(F).all() else ['(leadfield rows = picks)']
        F = np.asarray(F, float)                       # 约定: 行顺序与 picks 一致
    else:
        raise ValueError('leadfield has %d rows; expected either %d (one per '
                         'channel of raw) or %d (one per processed channel)'
                         % (F.shape[0], raw.info['nchan'], picks_good.size))
    if bad:
        raise ValueError('these channels have no lead field (they are probably '
                         'not in the forward solution): %s. Exclude them, e.g. '
                         'picks="meg".' % (bad,))

    # ------- 1) Gram 矩阵的 EVD -> 伪信号子空间投影 P (式(10)(11)) ------- #
    gamma, E = np.linalg.eigh(F @ F.T)                 # 升序
    gamma, E = gamma[::-1], E[:, ::-1]                 # 降序
    z = _select_nspace(Nspace, gamma, space_tol)
    if z >= M:
        warnings.warn('z = %d >= 通道数 %d: P 退化成单位矩阵, B_out 近似为零, '
                      'DSSP 将无法工作(图 20 的"钝切"性质消失)' % (z, M))
    P = E[:, :z] @ E[:, :z].T                          # (M x M)

    # ------------- 2) 里外两侧的数据矩阵 (式(20)(21)) ------------- #
    B_in = P @ B
    B_out = B - B_in

    # ------------- 3) 两边行空间的基 U, V (式(31)-(34)) ------------- #
    s_in, Vh_in = np.linalg.svd(B_in, full_matrices=False)[1:]
    s_out, Vh_out = np.linalg.svd(B_out, full_matrices=False)[1:]
    m = _select_ncomp(Nin, s_in, 'm (dimension of the inside row space)')
    n = _select_ncomp(Nout, s_out, 'n (dimension of the outside row space)')
    # 数值秩以外的那几个奇异向量是零空间里的任意向量, 一旦取进来就会制造出
    # 假的"公共成分"(cos theta ≈ 1), 所以这里按数值秩封顶。
    m_max = _effrank(s_in, rank_tol)
    n_max = _effrank(s_out, rank_tol)
    if m > m_max:
        warnings.warn('m = %d 超过了 B_in 的数值秩 %d, 已自动收缩(取进来的是零'
                      '空间的任意向量, 会造成假交集)' % (m, m_max))
        m = m_max
    if n > n_max:
        warnings.warn('n = %d 超过了 B_out 的数值秩 %d, 已自动收缩(同上)。'
                      'B_out 秩偏低通常说明 z 太大, 干扰几乎全被 P 投到了里侧'
                      % (n, n_max))
        n = n_max
    if m < 1 or n < 1:
        raise ValueError('B_in or B_out has (numerically) no rank: m_max = %d, '
                         'n_max = %d. Check Nspace (z): with z = M, P is the '
                         'identity and B_out vanishes.' % (m_max, n_max))
    print('Using z = %d, m = %d, n = %d' % (z, m, n))
    U = Vh_in[:m].T                                    # (K x m) 时间域基
    V = Vh_out[:n].T                                   # (K x n)

    # ------------- 4) 主角余弦与交集 (式(35)-(39)) ------------- #
    # U^T V = Y diag(cos(theta)) Z^T, 也可用 (V Z) 的前 r 列作基, 二者等价
    Y, cos_theta = np.linalg.svd(U.T @ V, full_matrices=False)[:2]
    r = _select_r(Nee, cos_theta, st_correlation)
    print('Using r = %d (cos(theta) threshold %g)' % (r, st_correlation))
    if r == 0:
        warnings.warn('没有找到 cos(theta) >= %g 的公共成分: 要么本来没有干扰, '
                      '要么干扰源在源空间内/边界上, 要么 Nspace 选得不合适。'
                      '数据将原样返回。' % st_correlation)
    G = (U @ Y)[:, :r]                                 # (K x r)

    # ------------- 5) 去掉干扰 (式(39)(40)) ------------- #
    B_clean = B - (B @ G) @ G.T                        # B (I - G G^T)

    raw_clean = raw.copy()
    raw_clean._data[picks_good, :] = B_clean

    if not return_diag:
        return raw_clean

    diag = dict(gamma=gamma, z=z, P=P, s_in=s_in, s_out=s_out, m=m, n=n,
                cos_theta=cos_theta, r=r, G=G, U=U, V=V,
                picks_good=picks_good, picks_bad=picks_bad, n_chan=M,
                n_times=K, sfreq=raw.info['sfreq'],
                st_correlation=st_correlation)
    return raw_clean, diag
