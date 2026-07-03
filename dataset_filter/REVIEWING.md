# PLAUSIBLE review — reviewer guide

You have ~200 counting images. For each, decide whether the dataset's **labeled count** is
believable (it is the number we are *auditing* — **not** trusted ground truth). ~15–20 min.

## 1. Open your shard — pick one

**A · Download & open (simplest, fully offline)**
In VS Code's Della file explorer, download your `review_<name>.html` and open it in any browser.

**B · Serve from Della + submit straight back**
```bash
ssh <netid>@della.princeton.edu
cd /scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs/full_sync/manual_review
python serve_review.py              # prints a localhost URL + the port to forward
```
In VS Code (Remote-SSH to Della): **Ports → Forward** the printed port → open the URL → click your name.

## 2. Review each card

- Count the objects yourself. Click the image to **zoom**; toggle **GT marks** to see where the
  annotation placed its points (magenta rings; shown only where the image has point annotations).
- **Keep** if the labeled count is believable · **Drop** if it's clearly wrong.
- Optional: type a **corrected count** and/or a note.
- Keys: `k` keep · `d` drop · `j`/`n` next unreviewed · `z` zoom.
- Progress auto-saves in your browser (`localStorage`) — refreshing is safe.

## 3. Send it back

- **Recipe B:** click **Submit to Della** → saved to `exports/review_<name>.json` automatically.
- **Recipe A:** click **Export JSON** → send the downloaded file to Chris for merging.
