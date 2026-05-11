# Leafsearch Project

## Align images for spatial histograms

The preprocessing pipeline now rotates each segmented leaf so the main leaf
axis is vertical before resizing to `256x256`.

Regenerate processed images from the raw dataset:

```bash
python scripts/realign.py
```

After changing processed images, rebuild the feature database:

```bash
python scripts/ingest.py --force
python -X utf8 scripts/normalize.py
```
