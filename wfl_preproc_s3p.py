# -*- coding: utf-8 -*-
"""
S3P (spectral signal space projection) —— 按论文复现, 适配 mne.io.Raw

论文
    R. R. Ramírez, B. H. Kopell, C. R. Butson, B. C. Hiner, S. Baillet,
    "Spectral signal space projection algorithm for frequency domain MEG and EEG
    denoising, whitening, and source imaging", NeuroImage 56 (2011) 78-92.

用法
    from wfl_preproc_s3p import s3p, pf_s3p

    # 标准 S3P: 每个频率都剔除 3 维噪声子空间
    raw_clean = s3p(raw, n_noise=3)

    # 逐频率指定维数 / 只看 0-250 Hz / 用空房间估计噪声子空间
    raw_clean = s3p(raw, raw_noise=raw_room, n_noise={60: 2, 120: 1},
                    fmax=250.)

    # pf-S3P: 全自动, 只削掉异常谱峰(工频及谐波等)
    raw_clean = pf_s3p(raw, fmax=250.)

    # 看诊断量: 各频率的功率、自动选出来的维数、特征值谱
    raw_clean, diag = s3p(raw, n_noise=3, return_diag=True)

注意
    * 投影是"逐频率、对全部时间帧"作用的(P~⊥(f) 与 τ 无关), 所以某频率上被
      去掉的空间成分在其它频率上会被保留 —— 这正是 S3P 频率特异性之所在。
    * 逆变换用加权重叠相加(WOLA); 无投影时(如 n_noise=0)可精确重构输入。
    * 一次投影后得到的谱一般不再是某个实信号的单边 STFT, 逆变换按最小二乘意义
      取实部(与 MATLAB 里 rfft/irfft 的常规做法一致)。
    * 窗函数的谱泄漏: 单个单频干扰在 STFT 里同时落在相邻的 ±1 个 bin 上(如 Hann
      窗: 主 bin 约 2/3 能量, 两侧各约 1/6)。所以只对"正好那个频率"投影只能去掉
      约 87%(实测: 只处理 60.00 Hz 时残余 12.4%; 连 59.75/60.25 Hz 一起处理后
      残余 0.26%)。要彻底去掉工频及其谐波, 可以: a) 整带指定维数(如 n_noise=1);
      b) 让 pf_freqs 覆盖邻近的 bin; c) 用 pf 自动模式。
    * pf-S3P 的判据是"该频率的功率高于邻域百分位就继续削", 因此很窄的脑活动谱峰
      (如尖锐的 alpha)也可能被判成异常峰而被削掉 —— 论文也特意提醒不要对感兴趣
      的窄带脑峰使用滑动频率窗, 必要时用 pf_freqs 限定在已知噪声频率上。
"""

import numpy as np

from mne.io.pick import _picks_to_idx


# --------------------------------------------------------------------------- #
# 窗函数 / 分帧 / STFT / 逆 STFT
# --------------------------------------------------------------------------- #
def _make_window(taper, n_win, kaiser_beta=8.6):
    taper = str(taper).lower()
    if taper in ('none', 'rect', 'boxcar', 'box'):
        return np.ones(n_win)
    if taper == 'hann':
        return np.hanning(n_win)
    if taper == 'hamming':
        return np.hamming(n_win)
    if taper == 'blackman':
        return np.blackman(n_win)
    if taper == 'kaiser':
        return np.kaiser(n_win, kaiser_beta)
    raise ValueError('unknown taper: %r' % (taper,))


def _frame_info(n_times, n_win, n_step):
    """返回 (左侧补零长度, 右侧补零长度, 各帧起点)。两端都补, 保证每个样本都被
    窗函数覆盖(避免 Hann 窗边缘为 0 导致重构时除零)。"""
    if n_times < n_win:
        raise ValueError('data length (%d) is shorter than the window (%d)'
                         % (n_times, n_win))
    pad = n_win // 2
    n_pad = n_times + 2 * pad
    n_frames = 1 + int(np.ceil((n_pad - n_win) / float(n_step)))
    end = (n_frames - 1) * n_step + n_win
    pad_right = pad + (end - n_pad)
    starts = np.arange(n_frames) * n_step
    return pad, pad_right, starts


def _stft_block(Xp, starts, n_win, window):
    """对若干帧做 STFFT。Xp: (n_ch, n_times_padded) -> (n_freqs, n_ch, n_frames)。"""
    n_ch = Xp.shape[0]
    n_freqs = n_win // 2 + 1
    Z = np.empty((n_freqs, n_ch, len(starts)), complex)
    buf = np.empty((n_ch, n_win), float)
    for i, s in enumerate(starts):
        np.multiply(Xp[:, s:s + n_win], window, out=buf, casting='unsafe')
        Z[:, :, i] = np.fft.rfft(buf, axis=-1).T
    return Z


def _istft_accum(Z, starts, n_win, window, acc, wsum):
    """加权重叠相加: acc += Σ_m w * irfft(Z_m), wsum += Σ_m w^2。"""
    for i, s in enumerate(starts):
        y = np.fft.irfft(Z[:, :, i].T, n=n_win, axis=-1)   # (n_ch, n_win), 实数
        y *= window
        acc[:, s:s + n_win] += y
        wsum[s:s + n_win] += window ** 2


def _frame_blocks(starts, frame_block):
    for i in range(0, len(starts), frame_block):
        yield starts[i:i + frame_block]


# --------------------------------------------------------------------------- #
# 参数处理
# --------------------------------------------------------------------------- #
def _align_channels(raw, raw_other):
    """保证另一段记录(如空房间)与 raw 的通道顺序一致。"""
    if list(raw_other.ch_names) == list(raw.ch_names):
        return raw_other
    if set(raw_other.ch_names) != set(raw.ch_names):
        raise ValueError('raw and raw_noise must contain the same channels')
    return raw_other.copy().reorder_channels(raw.ch_names)


def _resolve_n_noise(n_noise, freqs):
    """把 n_noise 统一成每个频点一个整数维数。

    n_noise 可以是: 标量 / 逐频率数组 / {频率: 维数} 字典(取最近的频率) /
    可调用对象 f->dim。
    """
    if callable(n_noise):
        out = np.asarray(n_noise(freqs), float)
    elif isinstance(n_noise, dict):
        keys = np.array(sorted(n_noise.keys()), float)
        vals = np.array([n_noise[k] for k in sorted(n_noise.keys())], float)
        idx = np.abs(freqs[:, None] - keys[None, :]).argmin(axis=1)
        out = vals[idx]
    elif np.isscalar(n_noise):
        out = np.full(freqs.shape, float(n_noise))
    else:
        out = np.asarray(n_noise, float)
        if out.shape != freqs.shape:
            raise ValueError('n_noise array must match the number of '
                             'analyzed frequencies')
    return np.maximum(np.round(out).astype(int), 0)


def _resolve_bandwidth(pf_bw, freqs):
    if pf_bw is None:
        return np.maximum(2.0, 0.1 * freqs)      # 默认带宽 ~ max(2 Hz, 10% f)
    if callable(pf_bw):
        return np.asarray(pf_bw(freqs), float)
    return np.full(freqs.shape, float(pf_bw))


def _pf_noise_dim(evals, n_base, freqs, percentile, pf_bw, pf_freqs,
                  pf_max_dim, nbins):
    """pf-S3P (式(23)): 自动确定额外剔除的维数 dυ(f)。

    evals : (n_freqs, n_ch) 降序特征值
    p(f)  : 邻域频段内"去掉 baseline 后总功率"的百分位(默认中位数)
    """
    n_ch = evals.shape[1]
    # baseline 去掉后每个频率剩下的总功率
    power = np.array([evals[i, n_base[i]:].sum() for i in range(evals.shape[0])],
                     float)
    bw = _resolve_bandwidth(pf_bw, freqs)
    pctl = np.empty_like(power)
    for i, f in enumerate(freqs):
        lo, hi = f - bw[i] / 2.0, f + bw[i] / 2.0
        sel = (freqs >= lo) & (freqs <= hi)
        pctl[i] = np.percentile(power[sel], percentile)

    n_extra = np.zeros(evals.shape[0], int)
    for i, f in enumerate(freqs):
        if pf_freqs is not None:
            # 只对用户指定的频率做 pf (论文: pf-S3P 可以只作用于已知噪声峰)
            if not np.any(np.abs(np.asarray(pf_freqs, float) - f) <= nbins):
                continue
        lam = evals[i]
        d_max = int(min(pf_max_dim, n_ch) - n_base[i])
        if d_max <= 0:
            continue
        best_val, best_d = None, 0
        for d in range(d_max + 1):
            val = abs(pctl[i] - lam[n_base[i] + d:].sum())
            if best_val is None or val < best_val:
                best_val, best_d = val, d
        n_extra[i] = best_d
    return n_extra, power, pctl


# --------------------------------------------------------------------------- #
# CSD 与特征分解
# --------------------------------------------------------------------------- #
def _csd_eig(Xp, starts, n_win, window, band, kmax, max_csd_bytes, frame_block):
    """逐频率估计 CSD 并做复特征分解。

    为了不让 (n_freqs, n_ch, n_ch) 的 CSD 占满内存, 频带按内存预算切成若干块,
    每块重新扫一遍时间帧(FFT 只算必要的几次)。
    """
    n_ch = Xp.shape[0]
    n_frames = len(starts)
    evals = np.zeros((band.size, n_ch))
    evecs = np.zeros((band.size, n_ch, min(kmax, n_ch)), complex)

    per_freq_bytes = max(n_ch * n_ch * 16, 1)      # complex128
    chunk = int(max(1, max_csd_bytes // per_freq_bytes))
    for c0 in range(0, band.size, chunk):
        sel = band[c0:c0 + chunk]
        csd = np.zeros((sel.size, n_ch, n_ch), complex)
        for blk in _frame_blocks(starts, frame_block):
            Z = _stft_block(Xp, blk, n_win, window)[sel]      # (k, n_ch, n_blk)
            csd += np.einsum('fcn,fkn->fck', Z, Z.conj())
        csd /= float(n_frames)                                # 式(10): 1/dτ
        for j in range(sel.size):
            lam, E = np.linalg.eigh(csd[j])                   # 式(11), 升序
            k = evecs.shape[2]
            evals[c0 + j] = lam[::-1]
            evecs[c0 + j, :, :k] = E[:, ::-1][:, :k]
    return evals, evecs


# --------------------------------------------------------------------------- #
# 主函数
# --------------------------------------------------------------------------- #
def s3p(raw, raw_noise=None, picks=None, n_noise=1, mode='noise',
        fmin=0.0, fmax=None, win_len=4.0, win_step=None, taper='hann',
        kaiser_beta=8.6, demean=True, restore_mean=True, pf=False,
        pf_percentile=50.0, pf_bw=None, pf_freqs=None, pf_max_dim=10,
        frame_block=8, max_csd_bytes=2 ** 28, return_diag=False):
    """S3P 去噪, 输入输出都是 mne.io.Raw。

    Parameters
    ----------
    raw : mne.io.Raw
        待处理的记录。
    raw_noise : mne.io.Raw | None
        用于估计 CSD / 噪声子空间的记录(空房间、静息段、伪迹段...)。为 None 时
        用 raw 自己估计(论文的默认做法)。
    picks : str | list | None
        参与处理的通道; 坏道保持原样。
    n_noise : int | array | dict | callable
        每个频率要剔除的噪声子空间维数(论文的 baseline noise dimension)。
        dict 形式如 {60.: 2, 120.: 1}(取最近频率), callable 形式如 lambda f: 1.
    mode : 'noise' | 'signal'
        'noise' : P = I - E_n E_n^H, 剔除噪声子空间(式(14), 默认)。
        'signal': P = E_n E_n^H, 只保留由高信噪比时段的 CSD 定义出的信号子空间
                  (式(7), 与前者等价, 取决于 CSD 从什么数据估计)。
    fmin, fmax : float
        只在这个频带内做投影(Hz); fmax=None 表示到 Nyquist。
    win_len, win_step : float
        STFFT 窗长与窗移(秒); win_step 默认 win_len/2(即 50% 重叠)。
    taper : str
        'hann'(默认) / 'kaiser'(论文用的) / 'hamming' / 'blackman' / 'none'。
    kaiser_beta : float
        taper='kaiser' 时的 beta。
    demean : bool
        是否先逐通道去均值(论文做法, 默认 True)。
    restore_mean : bool
        去噪后是否把均值加回去(默认 True, 让输出与输入的直流/偏置一致)。
    pf : bool
        是否启用 pf-S3P(式(23))自动确定额外维数。n_noise=0 + pf=True 即全自动。
    pf_percentile : float
        pf 用的百分位, 50 = 中位数。
    pf_bw : float | callable | None
        pf 的滑动频率窗宽度(Hz); None -> max(2 Hz, 0.1*f)。
    pf_freqs : array | None
        只在指定频率上做 pf(如 [60., 120.]) ; None 表示所有分析频率。
    pf_max_dim : int
        单频率总维数上限(同时决定需要保存多少特征向量, 默认 10)。
    frame_block : int
        每次处理的帧数(内存/速度折中)。
    max_csd_bytes : int
        CSD 数组的内存预算, 超过则把频带切块处理。
    return_diag : bool
        为 True 时额外返回诊断字典。

    Returns
    -------
    raw_clean : mne.io.Raw
    diag : dict (仅当 return_diag=True)
        freqs, eigvals, n_noise(最终维数), n_baseline, n_extra, power(pctl 前后),
        n_frames, n_win, n_step, mode 等。
        注意: 特征值/功率/自动维数都来自"用于估计 CSD 的那段记录"(raw_noise 或
        raw 本身)。想看去掉的分量, 直接相减即可: removed = raw - raw_clean。
    """
    # ----------------------------- 基本参数 ----------------------------- #
    sfreq = float(raw.info['sfreq'])
    n_win = int(round(float(win_len) * sfreq))
    if n_win < 4:
        raise ValueError('win_len too short')
    win_step = float(win_len) / 2.0 if win_step is None else float(win_step)
    n_step = int(round(win_step * sfreq))
    if n_step < 1:
        raise ValueError('win_step too short')
    if n_step > n_win:
        raise ValueError('win_step must be <= win_len')
    if mode not in ('noise', 'signal'):
        raise ValueError("mode must be 'noise' or 'signal'")
    window = _make_window(taper, n_win, kaiser_beta)

    n_freqs = n_win // 2 + 1
    freqs_all = np.fft.rfftfreq(n_win, d=1.0 / sfreq)
    f_hi = freqs_all[-1] if fmax is None else min(float(fmax), freqs_all[-1])
    band = np.where((freqs_all >= float(fmin)) & (freqs_all <= f_hi))[0]
    if band.size == 0:
        raise ValueError('no frequency bin in [fmin, fmax]')
    freqs = freqs_all[band]
    df = float(freqs_all[1] - freqs_all[0]) if n_freqs > 1 else 0.0

    # ----------------------------- 通道选择 ----------------------------- #
    picks = _picks_to_idx(raw.info, picks, exclude=())
    picks_good, picks_bad = list(), list()
    for ii, pi in enumerate(picks):
        if raw.ch_names[pi] in raw.info["bads"]:
            picks_bad.append(ii)
        else:
            picks_good.append(ii)
    picks_good = np.array(picks_good, int)
    picks_bad = np.array(picks_bad, int)
    if picks_good.size == 0:
        raise ValueError('no good channel to process')

    # ------------------------------- 数据 ------------------------------- #
    X = np.asarray(raw.get_data()[picks_good, :], float)      # (n_ch, n_times)
    n_ch, n_times = X.shape
    if demean:
        mean = X.mean(axis=1, keepdims=True)
        X = X - mean
    else:
        mean = np.zeros((n_ch, 1))

    pad, pad_right, starts = _frame_info(n_times, n_win, n_step)
    Xp = np.pad(X, ((0, 0), (pad, pad_right)), mode='constant')

    # 用于估计 CSD 的数据(默认就是 raw 自己)
    if raw_noise is None:
        Xn_pad, starts_n = Xp, starts
    else:
        raw_noise = _align_channels(raw, raw_noise)
        if abs(float(raw_noise.info['sfreq']) - sfreq) > 1e-6:
            raise ValueError('raw_noise must have the same sampling frequency')
        Xn = np.asarray(raw_noise.get_data()[picks_good, :], float)
        if demean:
            Xn = Xn - Xn.mean(axis=1, keepdims=True)
        pad_n, pad_r_n, starts_n = _frame_info(Xn.shape[1], n_win, n_step)
        Xn_pad = np.pad(Xn, ((0, 0), (pad_n, pad_r_n)), mode='constant')

    # --------------------- 每个频率的维数与特征分解 --------------------- #
    n_base = np.clip(_resolve_n_noise(n_noise, freqs), 0, n_ch)
    kmax = max(int(n_base.max()) if n_base.size else 0,
               int(pf_max_dim) if pf else 0, 1)
    kmax = int(min(kmax, n_ch))

    evals, evecs = _csd_eig(Xn_pad, starts_n, n_win, window, band, kmax,
                            int(max_csd_bytes), int(frame_block))

    n_extra = np.zeros(band.size, int)
    power = np.array([evals[i, n_base[i]:].sum() for i in range(band.size)],
                     float)
    pctl = np.full(band.size, np.nan)
    if pf:
        n_extra, power, pctl = _pf_noise_dim(
            evals, n_base, freqs, float(pf_percentile), pf_bw, pf_freqs,
            int(pf_max_dim), 0.5 * df + 1e-9)
    n_sel = n_base + n_extra
    n_sel = np.clip(n_sel, 0, n_ch)

    # --------------------- 投影时频数据 + 逆变换 --------------------- #
    acc = np.zeros_like(Xp)
    wsum = np.zeros(Xp.shape[1])
    for blk in _frame_blocks(starts, frame_block):
        Z = _stft_block(Xp, blk, n_win, window)               # (n_freqs, n_ch, n_blk)
        for j, k in enumerate(band):
            n = int(n_sel[j])
            if n <= 0:
                continue
            E = evecs[j][:, :n]                               # (n_ch, n)
            if mode == 'noise':
                # P~⊥(f) B~(f) = B~(f) - E_n (E_n^H B~(f))        式(14)(15)
                Z[k] -= E @ (E.conj().T @ Z[k])
            else:
                Z[k] = E @ (E.conj().T @ Z[k])                # 式(7)
        _istft_accum(Z, blk, n_win, window, acc, wsum)

    X_clean = acc / np.maximum(wsum, np.finfo(float).tiny)
    X_clean = X_clean[:, pad:pad + n_times]
    if restore_mean:
        X_clean = X_clean + mean

    raw_clean = raw.copy()
    raw_clean._data[picks_good, :] = X_clean

    if not return_diag:
        return raw_clean

    power_after = np.array(
        [evals[i, n_sel[i]:].sum() for i in range(band.size)], float)
    diag = dict(
        freqs=freqs, band=band, eigvals=evals, n_noise=n_sel,
        n_baseline=n_base, n_extra=n_extra, power=power, pctl=pctl,
        power_after=power_after, n_frames=len(starts), n_win=n_win,
        n_step=n_step, n_chan=n_ch, sfreq=sfreq, mode=mode,
        taper=taper, picks_good=picks_good, picks_bad=picks_bad,
    )
    return raw_clean, diag


def pf_s3p(raw, **kwargs):
    """pf-S3P 便捷包装: 默认 baseline 维数 0(全自动只削异常谱峰)。"""
    kwargs.setdefault('pf', True)
    kwargs.setdefault('n_noise', 0)
    return s3p(raw, **kwargs)
