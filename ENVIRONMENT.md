# Environment

The code supports Python 3.10 or newer.

Core packages:

- NumPy 1.24 or newer
- PyYAML 6.0 or newer
- PyTorch 2.1 or newer
- Skyfield 1.45 or newer and sgp4 2.23 or newer for LEO ephemerides
- PyTorch Geometric 2.5 or newer and einops 0.7 or newer for satellite graph
  models

Install both platforms from the repository root:

```bash
python -m pip install -e "code/satellite[paper]"
python -m pip install -e code/uav
```

CPU execution is supported by both implementations. CUDA can be selected from
the training and evaluation entry points when a compatible PyTorch build is
available.

S4 and Mamba2 are exploratory temporal ablations. They use the official
upstream implementations and are imported only when selected. Install the
upstream S4 repository for `models.s4.s4.S4Block` and a compatible
`mamba_ssm` package for Mamba2.

