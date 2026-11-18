# Who pushes where (read before your first push)

## Folder ownership
| Person | Push ONLY to |
|---|---|
| **Monty** | `pathsnap/`, `scripts/`, `configs/`, `evaluation/hmm_eval.py`, `README.md`, `requirements.txt`, `.gitignore`, `docs/`, `results/` (candidate + HMM outputs) |
| **Sanchita** | `api/`, `models/common.py`, `models/train_lgbm.py`, `models/tune_lgbm_optuna.py`, `evaluation/metrics.py`, `artifacts/`, `results/` (harness, LightGBM, Optuna outputs) |
| **Sheetal** | `ui/`, `models/train_rf.py`, `models/train_mlp.py`, `evaluation/compare_models.py`, `report/REPORT.md` + final PDF, `results/plots/` (comparison plots, MLP loss) |
| **Patil** | `eda/`, `viz/`, `models/train_logreg.py`, `models/train_knn.py`, `models/train_gnb.py`, `results/eda/`, `results/maps/` |
| **Everyone** | your own file(s) in `report/sections/` |

Need a change in someone else's folder? **Ask them in the chat**, don't edit it yourself.
Need a new package in `requirements.txt`? Tell Monty.

## Every time you push
```bash
git pull --rebase origin main            # get everyone's latest work first
git add <only your files>                 # avoid "git add ." unless you've checked git status
git commit -m "eda: add gap analysis"     # prefix = your area (eda, ui, api, models, hmm, report...)
git push origin main
```
If `git pull --rebase` reports a conflict in a file you don't own, stop and ping Monty.

## Never commit
- anything in `data/` (raw, processed or evaluator trips)
- any file over 100 MB
- passwords, tokens or `.env`
- hardcoded trip IDs or answers in code

## First-time setup
1. Monty adds you as a collaborator (GitHub → Settings → Collaborators). Accept the email invite.
2. `git clone https://github.com/smonty-19/PathSnap_7_H.git`
3. `python -m venv .venv`, activate it, then `pip install -r requirements.txt`
4. Copy the dataset into `data/raw/` (see `data/README.md`).
5. Read `docs/team/00_TEAM_PLAN.md`, then your own file in `docs/team/`.
