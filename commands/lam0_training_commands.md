# Lambda-zero baseline commands

Run these commands from the repository root.

## Environment

```cmd
conda activate lander310
```

```cmd
cd /d "C:\Users\rabiei\My Research\reduced-order-lunar-lander"
```

```cmd
set RUNS=C:\Users\rabiei\My Research\reduced_order_lunar_lander_runs
```

## Cold training

```cmd
python ".\examples\reduced_order_lunar_lander\train_reduced_order_lunar_lander.py" --mode train --variation_lambda 0.0 --run_name "reduced_order_lander_lam0_theta20_step002_500k" --save_dir "%RUNS%" --total_steps 491520 --num_envs 8 --steps_per_rollout 4096 --update_epochs 5 --minibatch_size 128 --learning_rate 1e-4 --target_kl 0.03 --theta_limit_deg 20 --init_std_main 0.60 --init_std_theta 0.50 --final_std_main 0.10 --final_std_theta 0.05 --eval_every_rollouts 5 --eval_episodes 20 --final_eval_episodes 50 --step_penalty 0.02 --timeout_penalty 0
```

## Warm fine-tuning

```cmd
set COLD_RUN=%RUNS%\reduced_order_lander_lam0_theta20_step002_500k_lam0_20260527_232344
```

```cmd
python ".\examples\reduced_order_lunar_lander\train_reduced_order_lunar_lander.py" --mode train --variation_lambda 0.0 --run_name "reduced_order_lander_lam0_theta20_step002_warm" --save_dir "%RUNS%" --warm_start_path "%COLD_RUN%\final_model.pt" --total_steps 300000 --num_envs 8 --steps_per_rollout 4096 --update_epochs 4 --minibatch_size 256 --learning_rate 2e-5 --target_kl 0.012 --theta_limit_deg 20 --init_std_main 0.15 --init_std_theta 0.08 --final_std_main 0.06 --final_std_theta 0.03 --eval_every_rollouts 3 --eval_episodes 50 --final_eval_episodes 100 --step_penalty 0.02 --timeout_penalty 0
```

## Final 333-seed evaluation

```cmd
set WARM_RUN=%RUNS%\reduced_order_lander_lam0_theta20_step002_warm_lam0_20260527_233243
```

```cmd
python ".\examples\reduced_order_lunar_lander\train_reduced_order_lunar_lander.py" --mode eval --checkpoint "%WARM_RUN%\final_model.pt" --save_dir "%RUNS%" --final_eval_episodes 333 --theta_limit_deg 20 --step_penalty 0.02 --timeout_penalty 0
```
