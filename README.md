# Muharaf CNN–BiLSTM–CTC: reproducible study on historical Arabic handwriting

Code, line identifiers and instructions for the paper
**"What Limits CTC-Based Recognition of Historical Arabic Handwriting? An Error Analysis and Controlled Ablation on Muharaf-public"**
(W. Fakhet, S. El Khediri, S. Zidi).

A compact CNN–BiLSTM–CTC line recogniser (11.6 M parameters, trained from scratch, no synthetic pre-training)
is used as a reference system for a documented filtering protocol, a quantitative error analysis and a controlled ablation study
on [Muharaf-public](https://huggingface.co/datasets/aamijar/muharaf-public).

| Test set (1 235 lines) | CER (%) | WER (%) |
|---|---|---|
| Greedy decoding (3 seeds) | 8.67 ± 0.08 | 28.72 ± 0.40 |
| Prefix beam search + character 6-gram LM (3 seeds) | 7.88 ± 0.03 | 24.42 ± 0.32 |

Scores keep diacritics and use the filtered partition described below. They must not be read as a ranking against other
papers that use different splits or normalisation (see the paper, Section 8).

## Repository content

```
notebooks/
  muharaf_main_runs.ipynb     main runs (80 epochs), decoding, LM, validation error analysis (Tables 5-12)
  muharaf_ablation.ipynb      ablation study and decoding ablation (Tables 13-14)
scripts/
  export_line_ids.py          rebuilds the line-ID lists and the filter counts (Table 1) from the public dataset
data/line_ids/
  train_kept.txt  val_kept.txt  test_kept.txt   Hugging Face row indices of the kept lines
  filter_counts.json                            Table 1
  charset_muharaf_train.txt                     99 characters of the kept training transcriptions
results/                      per-run results (CSV/JSON) behind Tables 6 and 13
checkpoints/                  EMA weights (best.pt) of the released runs, or a link to them
```

The images and transcriptions of Muharaf are **not** redistributed here (CC BY-NC-SA 4.0).

## Data and partition

Dataset: `aamijar/muharaf-public` on the Hugging Face Hub, used with its released train / validation / test partitions
(22 092 / 1 069 / 1 334 lines). The partitions carry no page or writer identifier.

A line is kept if its normalised transcription (Unicode NFKC, bidirectional and zero-width characters removed,
whitespace collapsed, diacritics kept) contains at least one Arabic letter, no Latin letter, and at least 50 % Arabic characters
among its non-space characters. Result: **20 558 / 981 / 1 235** lines.

```bash
pip install datasets
python scripts/export_line_ids.py --out data/line_ids
```

must print these counts (train: 819 Latin, 707 without Arabic letter, 7 below 50 % Arabic, 1 empty; see Table 1 of the paper).
The files `*_kept.txt` contain the row index of each kept line in the corresponding split of the Hugging Face dataset
(`train`, `validation`, `test`).

## Installation

Experiments were run in Kaggle notebooks on one NVIDIA T4 (15 GB), with PyTorch 2.11 (CUDA 12.8) and timm 1.0.
Other dependencies: `datasets`, `rapidfuzz`, `scipy`, `numpy`, `Pillow`, `matplotlib`.
Exact versions: see `requirements.txt` (generate it from the environment used for the final runs: `pip freeze`).

## Reproducing the results

Both notebooks run on Kaggle (one T4) and download Muharaf-public automatically. Run the definition cells in order.

**Main runs and error analysis** (`notebooks/muharaf_main_runs.ipynb`). Set `cfg.seed` (1, 42 or 200) and run the notebook:
sanity check, training, final evaluation (greedy, prefix beam of 10, beam + 6-gram LM), then the validation diagnostics
(sections 12b to 12d: dot errors, orthographic variants, S/D/I operations, frame budget T/L), which give the figures of Tables 8 to 12.
Keep `DIAG_SPLIT = "val"`: the diagnostics of the paper are computed on the validation set.

**Reference configuration**: CNN from scratch, height 128, maximum width 2720, horizontal stride 4,
3-layer BiLSTM (384 units), 80 epochs, batch 16, AdamW (lr 1e-3, weight decay 1e-2), 3 warm-up epochs, cosine decay,
EMA 0.999, gradient clipping 5. The checkpoint is selected on validation CER; the language-model weights (alpha in {0.3, 0.6},
beta in {0.5, 1.5}) are selected on validation; the test set is evaluated once per run.
One 80-epoch run takes about 6.0 to 6.5 h on a T4 (about 3.5 GB of GPU memory).

**Ablations** (`notebooks/muharaf_ablation.ipynb`). Run the definition cells (sections 1 to 8), then `smoke_test()` once, then `run_queue()`.
The ablation runs of the paper use a 40-epoch schedule: set `EPOCHS = 40`. Reference and `aug_none` use seeds 1, 42, 200;
`aug_geom` uses seeds 42 and 200; `stride2`, `stride8`, `h96` and `rasm03` use seed 1 only.
Section 12 of the notebook aggregates the runs (mean, paired differences, LaTeX table).
The same notebook with `EPOCHS = 80`, `SEEDS = [1, 42, 200]` and `VARIANTS = ["ref"]` gives paired reference runs at the full schedule.

**Decoding ablation (Table 14).** `decoding_ablation()` needs no training. It runs on the `ref` checkpoints, or on any checkpoint listed in `DECODING_CKPTS`.

## Running on several Kaggle sessions or accounts

A Kaggle session is limited to 12 h, and one 80-epoch run takes about 6 to 6.5 h. The runs of the paper were therefore distributed over
several Kaggle sessions and accounts (one seed or one variant per notebook), with the same code and the same GPU type (T4);
only the seed and the output directory differ between runs. Per-epoch times differ slightly between sessions
(for instance 265 to 273 s for seeds 1 and 42, 288 to 302 s for seed 200), which is why training times in the paper are only indicative.

To reproduce this setting, or to merge runs made in different sessions:

1. In each session, set `SEEDS` and `VARIANTS` for the runs to do and call `run_queue()`; interrupted runs resume from `last.pt`.
2. At the end of each session, call `backup_results()` and download `ablation_backup.zip`
   (all `results.json` and `history.json`, plus `best.pt` of the variants listed in `extra_best`).
3. Unzip all the archives into the same folder tree `OUT_ROOT/ep{EPOCHS}/{variant}/seed{seed}/`.
4. Run section 12 (`load_results()`, `summarize()`): it gathers every `results.json` found under `OUT_ROOT` and computes the means and the paired differences,
   whatever the session that produced each run.

## Known differences between runs (please read)

* **Vocabulary of the three main runs.** The three main runs of Table 6 were produced with an earlier version of the code, whose output layer had
  105 character classes plus the CTC blank (the character inventory was built together with another dataset). The six additional symbols never occur as
  Muharaf training targets. Both notebooks released here build the vocabulary from the Muharaf training lines only (99 classes).
  The reference run with the final code (seed 1, 80 epochs) gives test CER 8.72 % / WER 28.46 % (greedy) and 7.92 % / 24.46 % with the 6-gram LM,
  inside the range of the three main runs. To reuse the exact 105-class inventory of an earlier checkpoint, set `cfg.extra_vocab_file`
  (main notebook) or `VOCAB_FROM_CKPT` (ablation notebook).
* **Augmentation seed.** In the main runs the random generator of the augmentations was not seeded, so those runs are not bit-wise reproducible.
  The ablation notebook seeds the augmentations. Expect results within the seed-to-seed spread reported in the paper (about 0.1 CER point), not identical numbers.
* **Unused options.** The main notebook still contains a ConvNeXt V2 backbone option (`cfg.backbone = "convnext"`) that is not used in the paper.
  Some code comments are in French.

## Licences

* Code: MIT (see `LICENSE`).
* Muharaf-public data: CC BY-NC-SA 4.0 (see the dataset card). Trained checkpoints and any text derived from the data
  (for instance the character language model) inherit these non-commercial and share-alike conditions.
