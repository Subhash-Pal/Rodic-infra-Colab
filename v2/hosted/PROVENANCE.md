# Vendored from the hosted POC

These folders are copied **unchanged** from
`https://github.com/Subhash-Pal/rodic-infraai-gemini-resilient`, commit
`0c2d89b535352a152a7794de26760a7b81f1dcc5`:

- `src/` (model definition, config)
- `rodic_inference/` (CODEBRIM + SDNET2018 inference, evidence arbitration)
- `rodic_vlm/` (Gemini visual-evidence extraction with retry / fallback)
- `app/*.json` (Gemini config and contracts)

The two checkpoints (`best_codebrim.pt`, `best_sdnet.pt`, 45 MB each) are NOT stored in this repository.
`v2/deploy_assets.py` downloads them on first start from the pinned commit above and verifies the SHA-256 in
`models/MANIFEST.json`. To keep this repo in sync after you change the hosted repo, re-copy the folders and run
`python v2/deploy_assets.py --update-manifest --from-dir <path to its models/>`.

Do not edit these files here; change the source repository and re-sync, otherwise the two copies drift.
