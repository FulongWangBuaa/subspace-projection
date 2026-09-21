# Subspace-Projection Artifact Removal for OPM-MEG

Preprocessing methods for **OPM-MEG** measurements, ready to be plugged into an MNE-Python-based analysis pipeline.

| Module                | Method                                            | Also requires                          | Reference                            |
| --------------------- | ------------------------------------------------- | -------------------------------------- | ------------------------------------ |
| `wfl_preproc_ctsp.py` | **CTSP** common temporal subspace projection       | One artifact (control) recording       | Watanabe et al., EMBC 2013           |
| `wfl_preproc_dssp.py` | **DSSP** dual signal subspace projection           | Source-space lead field / `mne.Forward`| Sekihara et al., J. Neural Eng. 2016 |
| `wfl_preproc_s3p.py`  | **S3P** spectral signal space projection (incl. pf-S3P) | None (empty room optional)       | Ramírez et al., NeuroImage 2011      |

---

## Table of Contents

- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Quick start](#quick-start): [CTSP](#1-ctsp-common-temporal-subspace-projection) · [DSSP](#2-dssp-dual-signal-subspace-projection) · [S3P / pf-S3P](#3-s3p--pf-s3p-spectral-signal-space-projection)
- [References](#references)

---

## Requirements

```bash
pip install mne numpy scipy matplotlib
```

- Python ≥ 3.9, MNE-Python ≥ 1.0 (uses internal APIs such as `mne.io.pick._picks_to_idx` and `mne.fixes._safe_svd`; watch out when upgrading across MNE major versions)

## Quick Start

### 1) CTSP — Common Temporal Subspace Projection

> **Idea**: acquire a second recording with the same stimulus but with the stimulation electrode moved a few centimetres away, which yields a control measurement `A` containing the artifact only. Intersect the **temporal subspaces** of the two recordings (the directions with `cosθ ≈ 1` are the common artifact components), then right-multiply the projection operator that maps the data onto the subspace orthogonal to it: `B_clean = B (I − U_r U_rᵀ)`.

```python
ctsp(raw, raw_room, picks=None, Nout=None, Nin=None, Nee=None,
     st_correlation=0.98, power_correction=True, return_diag=False)
```

- `Nin`=q, `Nout`=p: how many "distinctively large" temporal singular vectors each of the two recordings contributes
- `Nee`=r: dimension of the intersection; a value smaller than 1 is treated as a cosθ threshold
- **Two branches**: if `Nin`, `Nout` and `Nee` are all supplied → paper branch (truncated SVD, Eqs. (6)–(8)); otherwise → `st_correlation` branch (`cosθ ≥ st_correlation` determines r automatically; QR implementation in the MNE tSSS style). Supplying only one or two of the dimensions raises a warning and falls back to the latter branch
- `power_correction=True`: first apply the per-channel power correction of Sec. III.B of the paper (least-squares matching of the amplitude of `A` to `B`)

```python
raw_clean, diag = ctsp(raw, raw_room, Nin=2, Nout=5, Nee=2, return_diag=True)
print(diag['cos_theta'])      # inspect the principal angles to judge whether r is reasonable
```

**Important**: temporal subspaces can only be intersected on a shared time axis, so `raw` and `raw_room` must have the **same sampling rate and the same number of samples**. If their lengths differ, the function raises an error and tells you to crop first.

### 2) DSSP — Dual Signal Subspace Projection

> **Idea**: exploit the dual definition of the signal subspace in the spatial and temporal domains — **no separate artifact recording is needed**. Discretise the source space into voxels and assemble the augmented lead field matrix `F = [L(r₁) … L(r_N)]`, then eigendecompose the Gram matrix `FFᵀ` to obtain the spatial-domain pseudo signal subspace projector `P`; next split the data into an inside and an outside part, `B_in = P·B` and `B_out = (I−P)·B`. Because of the "dull" cutoff property of `P`, interference from **outside** the source space appears on both sides, so the intersection of the two **row spaces (temporal subspaces)** is the interference subspace `S_I`; finally right-multiply by `(I − GGᵀ)` to remove it.

```python
import mne
subjects_dir, subject = ***, ***  # replace with the values actually used
trans = mne.transforms.Transform('head', 'mri')
src = mne.setup_source_space(subject, spacing='oct6', add_dist='patch', subjects_dir=subjects_dir)
bem = mne.make_bem_solution(mne.make_bem_model(subject=subject, ico=4,
                                               conductivity=(0.3,), subjects_dir=subjects_dir))
fwd = mne.make_forward_solution(raw_filt.info, trans=trans, src=src, bem=bem,
                                meg=True, eeg=False, mindist=5.0, n_jobs=1)
leadfield = fwd['sol']['data']

dssp(raw, leadfield, picks=None, Nspace=None, Nin=20, Nout=20, Nee=None,
     st_correlation=0.99, space_tol=1e-3, rank_tol=1e-8, return_diag=False)
```

- `leadfield`: an `(n_channels, D)` matrix (the paper's `F`) or an `mne.Forward` (its gain matrix is taken automatically)
- `Nspace`=z: how many of the "clearly large" eigenvalues of the Gram matrix to keep; the paper gives no quantitative criterion — pass an integer, `'interactive'` (click the knee of the log10 eigenvalue curve), `'auto'`, or `None` (prompt on the command line)
- `Nin/Nout`=m, n: the paper recommends erring on the large side; all their real-data runs used 20 (the default, insensitive). Values beyond the effective rank are shrunk automatically with a warning
- `Nee`=r: `None`/`'auto'` uses `cosθ ≥ 0.99`; as measured in the paper: r=6 (simulation), r=1 (VNS patient MEG), r=3 (real SCEF data)


### 3) S3P/pf-S3P — Spectral Signal Space Projection
> **Idea**: the spatial patterns of many noise sources are **frequency-dependent** (power-line noise and its harmonics, environmental vibrations, digital watches, …), so the projection operator should also be designed per frequency:
short-time FFT → estimate the cross-spectral density (CSD) matrix `Σ(f) = B̃(f)B̃ᴴ(f)/dτ` at each frequency → complex eigendecomposition → build the **frequency-specific** complex projection operator `P̃⊥(f) = I − EₙEₙᴴ` from the first n(f) eigenvectors → apply it to the time–frequency data `B̃⊥(f) = P̃⊥(f)B̃(f)` → inverse-transform back to the time domain.
The difference from FD-SSP is that it uses the full complex eigenvectors and acts per frequency in the time–frequency domain.

```python
s3p(raw, raw_noise=None, picks=None, n_noise=1, mode='noise',
    fmin=0.0, fmax=None, win_len=4.0, win_step=None, taper='hann',
    kaiser_beta=8.6, demean=True, restore_mean=True,
    pf=False, pf_percentile=50.0, pf_bw=None, pf_freqs=None, pf_max_dim=10,
    frame_block=8, max_csd_bytes=2**28, return_diag=False)

pf_s3p(raw, **kwargs)      # = s3p(..., pf=True, n_noise=0): fully automatic, trims outlying spectral peaks only
```

- `n_noise`: number of noise-subspace dimensions removed at each frequency; a scalar, a per-frequency array, a dict such as `{60.: 2, 120.: 1}`, or a callable `f -> dim` all work
- `raw_noise`: another recording used to estimate the CSD / noise subspace (empty room, resting segment, …); when `None`, `raw` itself is used (the paper's default)
- `fmin/fmax`: project only within this frequency band; bins outside the band are passed through unchanged. `win_len/win_step/taper`: STFFT parameters (the paper used a 4 s window, 2 s overlap and a Kaiser taper)
- `pf*`: percentile, sliding-window bandwidth and frequency limits of pf-S3P

```python
raw_clean = s3p(raw, raw_room, fmin=30, fmax=33, n_noise=1)   # process only 30-33 Hz
raw_clean = s3p(raw, n_noise=3)                               # remove 3 dimensions at every frequency
raw_clean = pf_s3p(raw, fmax=250.)                            # automatically suppress power-line noise and its harmonics
removed   = raw - raw_clean                                   # the removed component, for QC
```

## References

1. T. Watanabe, Y. Kawabata, D. Ukegawa, S. Kawabata, Y. Adachi, K. Sekihara. **Removal of Stimulus-Induced Artifacts in Functional Spinal Cord Imaging.** 35th Annual Int. Conf. of the IEEE EMBS, Osaka, 2013, pp. 3391–3394.
2. K. Sekihara, Y. Kawabata, S. Ushio, S. Sumiya, S. Kawabata, Y. Adachi, S. S. Nagarajan. **Dual signal subspace projection (DSSP): a novel algorithm for removing large interference in biomagnetic measurements.** J. Neural Eng. 13 (2016) 036007.
3. R. R. Ramírez, B. H. Kopell, C. R. Butson, B. C. Hiner, S. Baillet. **Spectral signal space projection algorithm for frequency domain MEG and EEG denoising, whitening, and source imaging.** NeuroImage 56 (2011) 78–92.

