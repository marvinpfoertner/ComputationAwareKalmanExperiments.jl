# Computation-Aware Kalman Filtering and Smoothing

Code for the experiments in the paper "Computation-Aware Kalman Filtering and Smoothing".

The implementation of the CAKF and CAKS can be found in [ComputationAwareKalman.jl](https://github.com/marvinpfoertner/ComputationAwareKalman.jl).

## How to run the ERA5 experiments?

1. Follow the instructions at https://cds.climate.copernicus.eu/how-to-api to install the `cdsapi` Python package.
2. Run `experiments/era5/data/download.py` to download the dataset.
3. Rename the NetCDF file downloaded by the script to `era5_t2m_2022.nc` (or create a symlink).
4. Run the experiment scripts `experiments/era5/{01_filter,02_smoother,03_metrics_ongrid}.jl {3,6,12,24}-{1,2,4,...}`.

## Citation

If you use this code, please cite our paper

```bibtex
@misc{Pfoertner2024CAKF,
  author = {Pf\"ortner, Marvin and Wenger, Jonathan and Cockayne, Jon and Hennig, Philipp},
  title = {Computation-Aware {K}alman Filtering and Smoothing},
  year = {2024},
  publisher = {arXiv},
  doi = {10.48550/arxiv.2405.08971},
  url = {https://arxiv.org/abs/2405.08971}
}
```