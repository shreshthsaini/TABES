# TABES toy-scale benchmark report

Toy MDM, CPU. Config: `/home/user/TABES/tabes/configs/toy.yaml`.

## Pareto sweep — arithmetic

| sampler | steps | accuracy | fwd NFE | bwd NFE | wall (s) |
|---|---|---|---|---|---|
| boe | 2 | 0.960 | 20 | 10 | 0.49 |
| confidence | 2 | 0.960 | 10 | 0 | 0.12 |
| entropy | 2 | 0.960 | 10 | 0 | 0.12 |
| margin | 2 | 0.960 | 10 | 0 | 0.13 |
| random | 2 | 0.957 | 10 | 0 | 0.13 |
| boe | 4 | 0.960 | 30 | 15 | 0.57 |
| confidence | 4 | 0.960 | 15 | 0 | 0.14 |
| entropy | 4 | 0.960 | 15 | 0 | 0.13 |
| margin | 4 | 0.960 | 15 | 0 | 0.14 |
| random | 4 | 0.957 | 15 | 0 | 0.14 |
| boe | 8 | 0.960 | 30 | 15 | 0.62 |
| confidence | 8 | 0.960 | 15 | 0 | 0.13 |
| entropy | 8 | 0.960 | 15 | 0 | 0.13 |
| margin | 8 | 0.960 | 15 | 0 | 0.13 |
| random | 8 | 0.957 | 15 | 0 | 0.14 |

## Pareto sweep — sudoku4

| sampler | steps | accuracy | fwd NFE | bwd NFE | wall (s) |
|---|---|---|---|---|---|
| boe | 2 | 0.230 | 20 | 10 | 0.54 |
| confidence | 2 | 0.240 | 10 | 0 | 0.13 |
| entropy | 2 | 0.223 | 10 | 0 | 0.12 |
| margin | 2 | 0.220 | 10 | 0 | 0.13 |
| random | 2 | 0.233 | 10 | 0 | 0.14 |
| boe | 4 | 0.380 | 40 | 20 | 1.16 |
| confidence | 4 | 0.303 | 20 | 0 | 0.25 |
| entropy | 4 | 0.293 | 20 | 0 | 0.23 |
| margin | 4 | 0.310 | 20 | 0 | 0.23 |
| random | 4 | 0.293 | 20 | 0 | 0.24 |
| boe | 8 | 0.460 | 80 | 40 | 2.19 |
| confidence | 8 | 0.443 | 40 | 0 | 0.56 |
| entropy | 8 | 0.447 | 40 | 0 | 0.57 |
| margin | 8 | 0.443 | 40 | 0 | 0.53 |
| random | 8 | 0.400 | 40 | 0 | 0.54 |

## Ablations — arithmetic

| variant | accuracy | wall (s) |
|---|---|---|
| boe (full) | 0.960 | 0.56 |
| boe -TIS | 0.960 | 0.15 |
| boe -anti-collapse | 0.960 | 0.55 |
| boe -AQA (full backward) | 0.960 | 0.55 |
| boe rho=0.125 | 0.960 | 0.60 |
| boe rho=0.25 | 0.960 | 0.57 |
| boe rho=0.5 | 0.960 | 0.55 |

## Ablations — sudoku4

| variant | accuracy | wall (s) |
|---|---|---|
| boe (full) | 0.460 | 2.16 |
| boe -TIS | 0.443 | 0.51 |
| boe -anti-collapse | 0.460 | 2.18 |
| boe -AQA (full backward) | 0.457 | 2.04 |
| boe rho=0.125 | 0.470 | 2.18 |
| boe rho=0.25 | 0.477 | 2.19 |
| boe rho=0.5 | 0.460 | 2.16 |
