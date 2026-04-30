# Zero-Shot Transfer from Simple to Full Lunar Lander

This repository contains code and results for studying zero-shot transfer from a simplified/reduced Lunar Lander model to the full Lunar Lander model.

The main idea is to train a reduced-action policy in a simpler model, then evaluate that policy in the full model without additional training. The project also compares transfer behavior across several regimes and controllers.

## Project contents

- `src/` — training, transfer evaluation, common-metrics, and plotting scripts
- `models/reduced_policy_20260421_212827/` — final reduced/simple-model policy
- `results/zero_shot_transfer_158_seeds/` — common-seed evaluation results over 158 seeds
- `logs/` — training/evaluation logs

## Main model

The selected reduced policy is stored in:

`models/reduced_policy_20260421_212827/`

Important files:

- `best_model.pt`
- `best_obsnorm.npz`
- `best_metrics.json`
- `run_config.txt`

Original run folder:

`reduced_lander_safe0_lamlr0_lr5e-06_stdanneal_nenv8_warm_norm1_seed42_20260421_212827`

## Main scripts

```text
src/train_reduced_lander_objectivefix.py
src/compare_simple_to_full_transfer.py
src/common_metrics_from_saved_trajectories.py
src/plot_common_success_controllers_only.py
src/lunar_lander.py
```

## Zero-shot transfer formulation

A policy is trained in the simplified/reduced Lunar Lander model:

```math
\pi_{\theta}^{\mathrm{red}} = \arg\max_{\pi_\theta} \; \mathbb{E}_{\tau \sim P_{\mathrm{red}}, \pi_\theta}
\left[ \sum_{t=0}^{T} \gamma^t r_t \right]
```

The trained reduced policy is then evaluated in the full Lunar Lander model without additional training:

```math
J_{\mathrm{full}}(\pi_{\theta}^{\mathrm{red}}) =
\mathbb{E}_{\tau \sim P_{\mathrm{full}}, \pi_{\theta}^{\mathrm{red}}}
\left[ \sum_{t=0}^{T} \gamma^t r_t \right]
```

This is a zero-shot transfer setting because the policy parameters are not updated during evaluation in the full model.

## Results

Final common-metrics results are stored in:

`results/zero_shot_transfer_158_seeds/`

This folder includes:

- `metrics_common.csv`
- `metrics_common.json`
- `with_original/`
- `without_original/`

The plots summarize performance across regimes such as nominal, windy, biased, delay, and hard settings.

## Overlay videos

Overlay comparison videos across deployment regimes are included in `media/videos/`.

- [Nominal regime overlay video](media/videos/official_compare_nominal_seed10230.mp4)
- [Biased regime overlay video](media/videos/official_compare_biased_seed10230.mp4)
- [Windy regime overlay video](media/videos/official_compare_windy_seed10230.mp4)
- [Hard regime overlay video](media/videos/official_compare_hard_seed10230_1.mp4)

These videos show qualitative side-by-side behavior of the lander across different deployment regimes.

## Goal

The goal of this project is to evaluate whether a policy trained in a simple/reduced Lunar Lander setting can transfer zero-shot to the full Lunar Lander setting, and how that transfer compares with alternative controllers and regime variations.