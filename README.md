# 子空间投影类伪迹去除工具箱 (Subspace-Projection Artifact Removal for OPM-MEG)

面向 **OPM脑磁图** 测量的预处理方法。它把三类「子空间投影」伪迹/干扰去除算法统一封装成
`mne.io.Raw` 进、`mne.io.Raw` 出的函数，可以直接插进基于 MNE-Python 的分析流程。

| 模块 | 方法 | 额外需要 | 参考论文 |
|---|---|---|---|
| `wfl_preproc_ctsp.py` | **CTSP** 公共时间子空间投影 | 一段伪迹(对照)测量 | Watanabe et al., EMBC 2013 |
| `wfl_preproc_dssp.py` | **DSSP** 对偶信号子空间投影 | 源空间引导场 / `mne.Forward` | Sekihara et al., J. Neural Eng. 2016 |
| `wfl_preproc_s3p.py` | **S3P** 谱域信号子空间投影 (含 pf-S3P) | 无(可选空房间) | Ramírez et al., NeuroImage 2011 |

三篇论文原文放在仓库根目录：`ctsp.pdf` / `dssp.pdf` / `s3p.pdf`。

---

## 目录

- [目录结构](#目录结构)
- [环境依赖](#环境依赖)
- [数据格式](#数据格式)
- [快速开始](#快速开始)
- [三个方法](#三个方法)：[CTSP](#1-ctsp-公共时间子空间投影) · [DSSP](#2-dssp-对偶信号子空间投影) · [S3P / pf-S3P](#3-s3p--pf-s3p-谱域信号子空间投影)
- [方法对比与选型](#方法对比与选型)
- [参数速查](#参数速查)
- [常见问题](#常见问题)
- [参考文献](#参考文献)

---


## 环境依赖

```bash
pip install mne numpy scipy matplotlib
```

- Python ≥ 3.9，MNE-Python ≥ 1.0（用到了 `mne.io.pick._picks_to_idx`、`mne.fixes._safe_svd` 这类内部接口，MNE 大版本升级时请留意）
- S3P 对长记录做了分块处理，内存占用可控（见 `frame_block` / `max_csd_bytes`）


## 快速开始

`main.py` 是完整的 cell 式流程（在 VS Code / Jupyter 里按 `# %%` 逐块运行）：

```python
# 1) 读取 .basedata → mne.io.Raw (64 磁强计 + Trigger)
# 2) 滤波: raw.filter(2, 45).notch_filter(50)
raw_filt      = raws[0].copy().filter(2, 45, fir_design='firwin').notch_filter(50)
raw_room_filt = raws[1].copy().filter(2, 45, fir_design='firwin').notch_filter(50)

# 3) CTSP: 用空房间那一套伪迹测量做对照; 注意两段要对齐且等长
from wfl_preproc_ctsp import ctsp
raw_ctsp = ctsp(raw_filt.copy().crop(tmin=0, tmax=raw_room_filt.times[-1]),
                raw_room_filt, Nin=2, Nout=5, Nee=2)

# 4) DSSP: 需要源空间导向场(BEM forward), 不需要伪迹段
import mne
subjects_dir, subject = r'D:\科研\MRI', 'wangfulong1'
trans = mne.transforms.Transform('head', 'mri')      # 单位变换: 做过配准请换成真实 trans
src = mne.setup_source_space(subject, spacing='oct6', add_dist='patch', subjects_dir=subjects_dir)
bem = mne.make_bem_solution(mne.make_bem_model(subject=subject, ico=4,
                                               conductivity=(0.3,), subjects_dir=subjects_dir))
fwd = mne.make_forward_solution(raw_filt.info, trans=trans, src=src, bem=bem,
                                meg=True, eeg=False, mindist=5.0, n_jobs=1)
from wfl_preproc_dssp import dssp
raw_dssp = dssp(raw_filt, fwd['sol']['data'], Nspace=5, picks='meg', Nee=3)

# 5) S3P: 在指定频带内逐频率做空间投影; 也可以只削工频/谐波
from wfl_preproc_s3p import s3p, pf_s3p
raw_s3p   = s3p(raw_filt, raw_room_filt, fmin=30, fmax=33, n_noise=1)
raw_clean = pf_s3p(raw_filt, fmax=250.)

# 6) 用平均后的诱发响应做前后对比
events = mne.find_events(raw_filt, stim_channel='Trigger')
evoked = mne.Epochs(raw_s3p, events, tmin=-0.2, tmax=1,
                    baseline=(-0.2, 0), preload=True).average()
```

所有函数都返回**新的** `Raw`（内部 `raw.copy()`），不会改动输入；坏道（`raw.info['bads']`）原样保留、不参与投影。

---

## 三个方法

记号：数据矩阵 `B`（M 个传感器 × K 个时间点），`B = B_S(信号) + B_I(干扰/伪迹) + B_e(噪声)`。

### 1) CTSP 公共时间子空间投影

> 论文 *Removal of Stimulus-Induced Artifacts in Functional Spinal Cord Imaging*, EMBC 2013，式(1)–(8)

**思路**：用「同一刺激、电极移开几厘米」再测一次，得到只含伪迹的对照测量 `A`。把两次测量的**时间子空间**
求交（`cosθ ≈ 1` 的方向就是公共伪迹成分），再右乘投影算子把数据投到与该子空间正交的方向上：
`B_clean = B (I − U_r U_rᵀ)`。

```python
ctsp(raw, raw_room, picks=None, Nout=None, Nin=None, Nee=None,
     st_correlation=0.98, power_correction=True, return_diag=False)
```

- `Nin`=q、`Nout`=p：两段各自取多少个「显著大」的时间奇异向量
- `Nee`=r：交集维数；给小于 1 的数值时按 cosθ 阈值处理
- **两个分支**：`Nin`、`Nout`、`Nee` 三个都给 → 论文分支（截断 SVD，式(6)–(8)）；否则 → `st_correlation` 分支
  （`cosθ ≥ st_correlation` 自动定 r，MNE tSSS 风格的 QR 实现）；只给其中一两个维数会告警并回落到后者
- `power_correction=True`：先按论文 III.B 节做逐通道功率校正（把 `A` 的幅度按最小二乘对齐到 `B`）

```python
raw_clean, diag = ctsp(raw, raw_room, Nin=2, Nout=5, Nee=2, return_diag=True)
print(diag['cos_theta'])      # 看主角余弦, 判断 r 是否合理
```

**必须注意**：时间子空间只有在同一条时间轴上才能求交，所以 `raw` 与 `raw_room` 必须**采样率相同、点数相同**。
长度不同时函数会直接报错并提示先裁：

```python
raw      = raw.copy().crop(tmin=0., tmax=0.05)
raw_room = raw_room.copy().crop(tmin=0., tmax=0.05)   # 采样率不同还需先 resample
```

### 2) DSSP 双信号子空间投影

> 论文 *Dual signal subspace projection (DSSP)*, J. Neural Eng. 13 (2016) 036007，式(1)–(40)

**思路**：利用信号子空间在空间域/时间域的对偶定义，**不需要单独的伪迹段测量**。先把源空间离散成体素、拼出
增广导向场 `F = [L(r₁) … L(r_N)]`，对 Gram 矩阵 `FFᵀ` 做特征分解得到空间域伪信号子空间投影 `P`；再把数据分成
里外两侧 `B_in = P·B`、`B_out = (I−P)·B` —— 由于 `P` 的「钝切」性质，源空间**外面**的干扰会同时出现在两侧，
于是两侧**行空间（时间域）的交集**就是干扰子空间 `S_I`，最后右乘 `(I − GGᵀ)` 把它去掉。

```python
dssp(raw, leadfield, picks=None, Nspace=None, Nin=20, Nout=20, Nee=None,
     st_correlation=0.99, space_tol=1e-3, rank_tol=1e-8, return_diag=False)
```

- `leadfield`：`(n_channels, D)` 矩阵（论文的 `F`）或 `mne.Forward`（自动取它的 gain matrix）
- `Nspace`=z：Gram 矩阵取前多少个「明显大」的特征值；论文没给定量准则，可给整数、`'interactive'`
  （点 log10 特征值曲线的拐点）、`'auto'` 或 `None`（命令行输入）
- `Nin/Nout`=m、n：论文建议「宁可取大」，实测一律 20（默认，不敏感）；超过有效秩会自动收缩并告警
- `Nee`=r：`None`/`'auto'` 时按 `cosθ ≥ 0.99` 自动定；论文实测：模拟 r=6、VNS 病人 MEG r=1、真实 SCEF r=3

```python
raw_clean, diag = dssp(raw, fwd, Nspace=5, picks='meg', Nee=3, return_diag=True)
print(diag['gamma'], diag['cos_theta'], diag['r'])
```


### 3) S3P / pf-S3P 谱域信号子空间投影

> 论文 *Spectral signal space projection algorithm for frequency domain MEG and EEG denoising, whitening, and source imaging*, NeuroImage 56 (2011) 78–92，式(8)–(15)，pf-S3P 见式(23)

**思路**：很多噪声的空间图样是**随频率变化**的（工频及其谐波、环境振动、数字手表……），所以投影算子也该逐频率设计：
短时 FFT → 每个频率估计交叉谱密度矩阵 `Σ(f) = B̃(f)B̃ᴴ(f)/dτ` → 复特征分解 → 用前 n(f) 个特征向量构成
**频率特异**的复投影算子 `P̃⊥(f) = I − EₙEₙᴴ` → 作用到时频数据 `B̃⊥(f) = P̃⊥(f)B̃(f)` → 逆变换回时域。
与 FD-SSP 的区别就在于它用的是完整复特征向量、且逐频率作用在时频域。

```python
s3p(raw, raw_noise=None, picks=None, n_noise=1, mode='noise',
    fmin=0.0, fmax=None, win_len=4.0, win_step=None, taper='hann',
    kaiser_beta=8.6, demean=True, restore_mean=True,
    pf=False, pf_percentile=50.0, pf_bw=None, pf_freqs=None, pf_max_dim=10,
    frame_block=8, max_csd_bytes=2**28, return_diag=False)

pf_s3p(raw, **kwargs)      # = s3p(..., pf=True, n_noise=0): 全自动只削异常谱峰
```

- `n_noise`：每个频率剔除的噪声子空间维数；标量、逐频率数组、`{60.: 2, 120.: 1}` 字典、或 `f -> dim` 可调用对象都行
- `raw_noise`：用来估计 CSD / 噪声子空间的另一段记录（空房间、静息段…）；`None` 时用 `raw` 自己估计（论文默认做法）
- `fmin/fmax`：只在该频带内投影，带外原样通过；`win_len/win_step/taper`：STFFT 参数（论文用 4 s 窗、2 s 重叠、Kaiser taper）
- `pf*`：pf-S3P 的百分位、滑窗带宽、限定频率等

```python
raw_clean = s3p(raw, raw_room, fmin=30, fmax=33, n_noise=1)   # 只处理 30-33 Hz
raw_clean = s3p(raw, n_noise=3)                               # 整带每频率剔 3 维
raw_clean = pf_s3p(raw, fmax=250.)                            # 自动削工频及谐波
removed   = raw - raw_clean                                   # 被去掉的分量, 用于 QC
```

**两个实测出来的坑**：

1. **窗函数谱泄漏**：单个单频干扰在 STFT 里会同时落在相邻 ±1 个 bin（Hann 窗主 bin 约 2/3 能量）。只对「正好那个频率」
   投影只能去掉约 87%（实测残余 12.4%）；连 `59.75 / 60 / 60.25 Hz` 一起处理后残余降到 0.26%。所以要彻底压工频及谐波，
   要么整带指定维数（`n_noise=1`），要么让 `pf_freqs` 覆盖邻近 bin，要么直接用 pf 自动模式。
2. **pf-S3P 的判据是「比邻域百分位高就继续削」**，所以很窄的脑活动谱峰（如尖锐的 alpha）也可能被判成异常峰削掉；
   论文也特意提醒不要把滑动频率窗用在感兴趣的窄带脑峰上，必要时用 `pf_freqs` 限定在已知噪声频率。

---


## 参数速查

| 方法 | 参数 | 含义 | 建议 |
|---|---|---|---|
| CTSP | `Nin` / `Nout` (q/p) | 两段各自取的时间奇异向量个数 | 取奇异值谱「明显大」的个数，可用 `'interactive'` 点曲线 |
| CTSP | `Nee` (r) | 公共（干扰）子空间维数 | `cosθ ≥ 0.999` 的个数，或直接给整数 |
| CTSP | `st_correlation` | cosθ 阈值（QR 分支） | 0.98 起；想更严格用 0.999 |
| DSSP | `Nspace` (z) | 伪信号子空间维数 | 看 `log10(gamma)` 的拐点，先 `return_diag=True` |
| DSSP | `Nin` / `Nout` (m/n) | 里外两侧行空间维数 | 20（论文值，不敏感，可放大） |
| DSSP | `Nee` (r) | 干扰子空间维数 | `None` 自动（`cosθ ≥ 0.99`），必要时手动指定 |
| S3P | `n_noise` | 每频率剔除的噪声维数 | 1~3；可用 dict 逐频率指定 |
| S3P | `fmin` / `fmax` | 处理频带 | 只压工频就把带卡在峰值附近，记得覆盖 ±1 bin |
| S3P | `win_len` / `win_step` / `taper` | STFFT 参数 | 4 s / 2 s / `'hann'`（论文用 Kaiser） |
| S3P | `pf_*` | pf-S3P 自动维数 | `pf_percentile=50`、`pf_bw=None`、必要时 `pf_freqs` |

三个函数都支持 `return_diag=True`，返回奇异值谱、主角余弦、自动维数、各频率功率等诊断量，
**建议先跑一次看曲线再定参数**。



## 参考文献

1. T. Watanabe, Y. Kawabata, D. Ukegawa, S. Kawabata, Y. Adachi, K. Sekihara. *Removal of Stimulus-Induced Artifacts in Functional Spinal Cord Imaging.* 35th Annual Int. Conf. of the IEEE EMBS, Osaka, 2013, pp. 3391–3394.
2. K. Sekihara, Y. Kawabata, S. Ushio, S. Sumiya, S. Kawabata, Y. Adachi, S. S. Nagarajan. *Dual signal subspace projection (DSSP): a novel algorithm for removing large interference in biomagnetic measurements.* J. Neural Eng. 13 (2016) 036007.
3. R. R. Ramírez, B. H. Kopell, C. R. Butson, B. C. Hiner, S. Baillet. *Spectral signal space projection algorithm for frequency domain MEG and EEG denoising, whitening, and source imaging.* NeuroImage 56 (2011) 78–92.


