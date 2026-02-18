# TABES

TABES: Trajectory-Aware Backward-on-Entropy Steering for Masked Diffusion Models.

[![arXiv](https://img.shields.io/badge/arXiv-2602.00250-b31b1b.svg)](https://arxiv.org/abs/2602.00250)
![Code](https://img.shields.io/badge/Code-Coming%20Soon-8a8a8a)

## Overview

TABES is a training-free guidance framework for masked diffusion language models.  
It improves generation by combining:

- forward confidence steering (adaptive entropy minimization), and
- backward trajectory steering (uncertainty-aware correction from reversed trajectory states).

The method is designed to avoid trajectory lock-in and improve final generation quality without retraining the base model.

## Project Page

Once GitHub Pages is enabled for this repo, the site will be available at:

- `https://shreshthsaini.github.io/TABES/`
- detailed explainer blog: `https://shreshthsaini.github.io/TABES/blog.html`

Local page sources live in `docs/`.

## Publish GitHub Pages

1. Commit and push:

```bash
git add README.md CITATION.cff docs
git commit -m "Add TABES project page and method blog"
git push origin master
```

2. In GitHub, open `Settings -> Pages`.
3. Set `Source` to `Deploy from a branch`.
4. Select branch `master` and folder `/docs`, then save.

## Repo Structure

- `docs/index.html`: landing page with paper links and highlights
- `docs/blog.html`: detailed method walkthrough with figures
- `docs/assets/css/styles.css`: shared styles
- `docs/assets/images/`: custom explanatory figures
- `CITATION.cff`: machine-readable citation metadata

## Citation Guide

If this paper or project page helps your work, please cite:

```bibtex
@article{luo2026tabes,
  title   = {TABES: Trajectory-Aware Backward-on-Entropy Steering for Masked Diffusion Models},
  author  = {Luo, Yanchen and Wang, Peisong and Liao, Tinglong and Tong, Shengbang and Zhuang, Yufei and Zhu, Minjun and Zhang, Hao and Cun, Xiaodong and Xu, Qiang},
  journal = {arXiv preprint arXiv:2602.00250},
  year    = {2026}
}
```

## License

Released under the MIT License. See `LICENSE`.
