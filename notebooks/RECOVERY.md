# Error and recovery log (sync and CI/CD notebooks)

Keep this file updated. Add a row whenever a new error appears and how it was fixed.

## Errors seen so far

| Date | Notebook | What you saw | Cause | Recovery | Status |
|---|---|---|---|---|---|
| 18 Nov | `Rodic_00_GitHub_Sync` (v0.1) | `AssertionError: WORK_DIR not found: /content/work` | Each Colab has its own runtime, so `/content/work` never existed in the sync notebook | Work folder moved to Google Drive (`My Drive/rodic_work`), created automatically; use `Rodic_00_GitHub_Sync_v2` | Fixed |
| 18 Nov | `Rodic_00_GitHub_Sync` | Fixes disappeared, repo got "Created using Colab" commits | An old open tab re-saved itself to GitHub and overwrote the fixed file | Renamed to `_v2`, added a version stamp (`Running: 00-sync v0.3`); close old tabs; open only via the badge | Fixed |
| 18 Nov | `Rodic_00_GitHub_Sync_v2` | Last cell: `Nothing changed - nothing to push.` | Not an error. The Drive `rodic_work` folder and `NOTEBOOK_FILES` were both empty, so nothing was staged | Put files in `My Drive/rodic_work`, or list notebook names in `NOTEBOOK_FILES` | Expected |
| 18 Nov | `Rodic_01_CICD_Pipeline` | Last push cell said `Workflows already up to date.` but `.github/workflows` was not on GitHub and Actions showed 0 runs | An earlier push failed (token without Workflows permission), the commit stayed in the Colab's local clone, and the cell only pushed when it had something new to commit | v0.2: the cell pushes whenever local is ahead, prints `PUSH FAILED` with a hint, and lists `.github/workflows` as seen on GitHub. Fix the token scope and re-run that cell | Fixed in v0.2, awaiting re-run |
| 18 Nov | `Rodic_01_CICD_Pipeline` v0.2 | `PUSH FAILED ... Permission to Subhash-Pal/Rodic-infra-Colab.git denied to Subhash-Pal ... 403` | The notebooks read the secret `GITHUB_TOKEN-gemini-resilient`, a token that most likely has no write access to this repo. Clone worked only because the repo is public. A workflow-scope problem would mention `workflow` instead | v0.3 (00 v0.4): prefers secret `GITHUB_TOKEN`, falls back to the old name, prints which secret and whose token it is. Make a fine-grained token for this repo only (Contents + Workflows read/write, Actions read), save it as `GITHUB_TOKEN`, rerun from the config cell | Fix shipped, awaiting new token |

## Quick recovery table

| Symptom | Likely cause | Recovery |
|---|---|---|
| First cell does not print `Running: 00-sync v0.3` | Stale copy of the notebook | Close the tab, reopen from the badge in `notebooks/README.md`. Never use File > Save a copy in GitHub from an old tab |
| `Nothing to stage yet` or `Nothing changed` | Empty work folder, empty `NOTEBOOK_FILES`, or files identical to the repo | Add files or notebook names, rerun from the top |
| `Notebook not found in Drive` | Wrong file name, or the notebook was opened from GitHub and never saved to Drive | Check the exact name in `My Drive/Colab Notebooks`; File > Save a copy in Drive first |
| `SecretNotFoundError` for `GITHUB_TOKEN` | Secret missing, or Notebook access is off for this notebook | Colab key icon > add `GITHUB_TOKEN` > turn on Notebook access (the older secret `GITHUB_TOKEN-gemini-resilient` has a different name) |
| `git clone` 401/403/404 | Token expired, wrong scope, or not granted this repo | Make a new fine-grained token: this repo only, Contents read/write, Workflows read/write, Actions read |
| Push refused: `workflow` scope | Token lacks Workflows permission | Add Workflows read/write to the token (needed by notebook 01) |
| Push rejected (non-fast-forward) | Remote has newer commits | Rerun the last cell (it pulls with rebase). If a rebase conflict remains: `!rm -rf /content/_repo`, then run from the top |
| `BLOCKED: possible secrets found` | A key or token is inside a staged file | Remove it, then revoke and re-issue that key; do not just delete the line |
| `BLOCKED: tests failed` | A staged test fails | Fix it, or set `RUN_TESTS = False` knowingly |
| Gemini 429 (quota) | Key's quota used up; free-tier limits are per Google Cloud project, so keys in one project share it | Switch to the next key (`GEMINI_API_KEY1` to `5`); wait for the quota window to reset; `GEMINI_API_KEY3` was exhausted on 17 Nov |
| A bad commit reached `main` | Wrong files pushed | `git revert <sha>` and push; history stays intact. Restore a file: `git checkout <sha> -- path` |
| A token appears in notebook output | Leaked in a printed line | Revoke it on GitHub at once, create a new one, clear the cell output |

## Rules that prevent most errors

1. Open each notebook only through its Open in Colab badge.
2. Run cells top to bottom; the last cell is always the push.
3. Start with `DRY_RUN = True`; switch to `False` only after the staged list looks right.
4. Never paste a key or token into a cell or into chat.
