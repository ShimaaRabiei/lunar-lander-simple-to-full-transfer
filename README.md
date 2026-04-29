\# Zero-Shot Transfer from Simple to Full Lunar Lander



This repository contains code and results for studying zero-shot transfer from a simplified/reduced Lunar Lander model to the full Lunar Lander model.



The main idea is to train a reduced-action policy in a simpler model, then evaluate that policy in the full model without additional training. The project also compares transfer behavior across several regimes and controllers.



\## Project contents



\- `src/` — training, transfer evaluation, common-metrics, and plotting scripts

\- `models/reduced\_policy\_20260421\_212827/` — final reduced/simple-model policy

\- `results/zero\_shot\_transfer\_158\_seeds/` — common-seed evaluation results over 158 seeds

\- `logs/` — training/evaluation logs



\## Main model



The selected reduced policy is stored in:



`models/reduced\_policy\_20260421\_212827/`



Important files:



\- `best\_model.pt`

\- `best\_obsnorm.npz`

\- `best\_metrics.json`

\- `run\_config.txt`



Original run folder:



`reduced\_lander\_safe0\_lamlr0\_lr5e-06\_stdanneal\_nenv8\_warm\_norm1\_seed42\_20260421\_212827`



\## Main scripts



```text

src/train\_reduced\_lander\_objectivefix.py

src/compare\_simple\_to\_full\_transfer.py

src/common\_metrics\_from\_saved\_trajectories.py

src/plot\_common\_success\_controllers\_only.py

src/lunar\_lander.py

