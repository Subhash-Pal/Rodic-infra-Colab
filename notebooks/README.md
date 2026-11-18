# notebooks/

Colab notebooks for syncing work to this repo and running CI/CD. Open in Colab:

| Notebook | Open |
|---|---|
| `Rodic_01_CICD_Pipeline.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Subhash-Pal/Rodic-infra-Colab/blob/main/notebooks/Rodic_01_CICD_Pipeline.ipynb) |
| `Rodic_00_GitHub_Sync_v2.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Subhash-Pal/Rodic-infra-Colab/blob/main/notebooks/Rodic_00_GitHub_Sync_v2.ipynb) |

Existing notebooks (moved here from the repo root):

| Notebook | Open |
|---|---|
| `Rodic_Edge_CV.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Subhash-Pal/Rodic-infra-Colab/blob/main/notebooks/Rodic_Edge_CV.ipynb) |
| `rodic_infraai_gemini_resilient.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Subhash-Pal/Rodic-infra-Colab/blob/main/notebooks/rodic_infraai_gemini_resilient.ipynb) |

## Order

0. Create a fine-grained GitHub token for this repo only (Contents and Workflows read/write, Actions read). Add it as the Colab Secret `GITHUB_KEY-Rodic-infra-Colab` with Notebook access on. Never commit it.
1. Run `Rodic_01_CICD_Pipeline.ipynb` once to add CI (lint, secret check, pytest) and the tag-based release workflow.
2. Run `Rodic_00_GitHub_Sync_v2.ipynb` with `DRY_RUN = True` and check what would be committed.
3. Run `Rodic_00_GitHub_Sync_v2.ipynb` with `DRY_RUN = False` after each work session. Write code to `/content/work`, and put a `### ` heading above each cell that says what it adds; those headings become the commit message and `CHANGELOG.md`.

Status: written and syntax-checked, not yet run end to end in Colab. The CI gates are basic (syntax and undefined-name lint, secret pattern check, synthetic-data tests); there is no model-accuracy or Jetson test yet.
