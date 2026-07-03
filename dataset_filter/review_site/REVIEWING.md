# PLAUSIBLE review — reviewer guide

You have ~200 counting images. For each, decide whether the dataset's **labeled count** is
believable (it is the number we are *auditing* — **not** trusted ground truth). ~15–20 min.

Your shard is **one self-contained HTML file** in this folder on Della:
`/scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs/full_sync/manual_review/`

| Reviewer | Your file |
|---|---|
| Chris Cai | `review_chris-cai.html` |
| Gabe Sarch | `review_gabe-sarch.html` |
| Yipeng Zhang | `review_yipeng-zhang.html` |
| Haoyang Wu | `review_haoyang-wu.html` |

## 1. Open your shard — pick one

**A · Someone sends you the file (no Della needed).** The coordinator can just send you your
`review_<name>.html` (Slack/email/Drive). Open it in any browser — everything's embedded.

**B · Download it yourself from Della.** In VS Code (Remote-SSH to Della), open the folder above
in the Explorer, **right-click your file → Download**, then open it in your browser. (Or from a
laptop terminal: `scp <netid>@della.princeton.edu:<folder>/review_<name>.html ~/Downloads/`.)

**C · Serve it from Della + submit straight back (no filename to remember).**
```bash
ssh <netid>@della.princeton.edu
cd /scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs/full_sync/manual_review
python serve_review.py              # prints a localhost URL + the port to forward
```
In VS Code (Remote-SSH to Della): **Ports → Forward** the printed port → open the URL → the hub
page lists everyone by name → **click your name**.

## 2. Review each card

- Count the objects yourself. Click the image to **zoom**; toggle **GT marks** to see where the
  annotation placed its points (magenta rings; shown only where the image has point annotations).
- **Keep** if the labeled count is believable · **Drop** if it's clearly wrong. *(this is the main call)*
- Optional **confident count**: enter a number **only if you're 100% sure** of the exact count (leave blank otherwise). A note is optional too.
- Keys: `k` keep · `d` drop · `j`/`n` next unreviewed · `z` zoom.
- Progress auto-saves in your browser (`localStorage`) — refreshing is safe.

## 3. Send it back

- **Recipe C (served):** click **Submit to Della** → saved to `exports/review_<name>.json` automatically.
- **Recipe A/B (opened a file):** click **Export JSON** → send the downloaded file to Chris for merging.
