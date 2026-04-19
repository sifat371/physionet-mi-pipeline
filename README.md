# physionet-mi-pipeline
Clean 4-class motor imagery trial extraction pipeline for the PhysioNet EEGMMIDB dataset.

---

## Papers This Pipeline Is Based On

This pipeline combines the data protocol from two papers:

**Trial count (84 trials per subject):**
> Xue Q, Song Y, Wu H, Cheng Y and Pan H (2024). Graph neural network based on brain inspired forward-forward mechanism for motor imagery classification in brain-computer interfaces. *Frontiers in Neuroscience*, 18:1309594.
> https://doi.org/10.3389/fnins.2024.1309594

**Subject exclusion (103 valid subjects):**
> Zheng et al. (2025). Motor imagery EEG classification via wavelet-packet synthetic augmentation and multi-branch spatio-temporal convolutional Transformer. *Frontiers in Neuroscience*, 19:1689647.
> https://doi.org/10.3389/fnins.2025.1689647

---

## Dataset Overview

- **Source:** [PhysioNet EEGMMIDB v1.0.0](https://physionet.org/content/eegmmidb/1.0.0/)
- **Original subjects:** 109
- **Valid subjects after exclusion:** 103
- **Channels:** 64 EEG (10-10 layout)
- **Sampling rate:** 160 Hz
- **Task:** 4-class imagined motor imagery only

### Excluded Subjects

Subjects `38, 88, 89, 92, 100, 104` are excluded due to inconsistent sampling rates or labeling errors.

### 4 Classes

| Label | Class | Source Runs | Annotation |
|-------|-------|-------------|------------|
| 0 | Left fist | R04, R08, R12 | T1 |
| 1 | Right fist | R04, R08, R12 | T2 |
| 2 | Both fists | R06, R10, R14 | T1 |
| 3 | Both feet | R06, R10, R14 | T2 |

Only **imagined movement** runs are used.
Baseline (R01, R02) and real movement runs (R03, R05, R07, R09, R11, R13) are discarded.

### Trial Structure

```
84 trials per subject = 4 classes × 3 runs × 7 trials per run
Total = 103 subjects × 84 trials = 8,652 trials
```

---

## Output

A single HDF5 file `raw_EEG_4class.h5`:

| Dataset | Shape | dtype | Description |
|---------|-------|-------|-------------|
| `data` | (8652, 640, 64) | float32 | Raw EEG — trials × samples × channels |
| `labels` | (8652,) | int32 | Class labels 0–3 |
| `subjects` | (8652,) | int32 | Subject ID |

```
(8652, 640, 64)
  │      │    └─ 64 channels
  │      └────── 640 samples = 4 s × 160 Hz
  └──────────── 8652 = 103 subjects × 84 trials
```

---

## Quickstart

### 1. Download the raw dataset

```bash
wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/
```

Organize into this structure:

```
data/
├── S001/
│   ├── S001R04.edf
│   ├── S001R06.edf
│   ├── S001R08.edf
│   ├── S001R10.edf
│   ├── S001R12.edf
│   └── S001R14.edf
├── S002/
└── ...
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Extract trials

```bash
python extract.py
```
Iterates over all 103 valid subjects and loads 6 EDF run files per subject (R04, R06, R08, R10, R12, R14). For each run, it reads the MNE annotations, maps each T1/T2 event to a class label, and slices a 4-second window (640 samples at 160 Hz) starting at the event onset. A maximum of 7 trials per class per run is enforced, yielding exactly 84 trials per subject. All trials, labels, and subject IDs are stacked into NumPy arrays and saved as a single compressed HDF5 file (`raw_EEG_4class.h5`).

Output: `raw_EEG_4class.h5`

### 4. Verify the output

```bash
python verify.py
```
Loads `raw_EEG_4class.h5` and runs 6 automated checks: array shape, class balance, subject balance, NaN/Inf detection, per-subject label distribution, and signal amplitude sanity. Channel names are read directly from the EDF header via MNE and stripped of BCI2000 padding dots, so C3/Cz/C4 lookups are always correct. On passing all checks, a plot of one raw trial per class across channels C3, Cz, and C4 is saved to `output/eeg_verification_plot.png` for visual inspection.

---

## Configurable Parameters

All key parameters are at the top of `extract.py`.

```python
DATA_PATH         = './data'               # path to your .edf files
OUTPUT_FILE       = './raw_EEG_4class.h5'  # output path
SR                = 160                    # sampling rate (Hz)
T_LEN             = 640                    # trial length in samples (4 s)
MAX_TRIALS        = 7                      # max trials per class per run
EXCLUDED_SUBJECTS = {38, 88, 89, 92, 100, 104}
TASK2_RUNS        = {'R04', 'R08', 'R12'}
TASK4_RUNS        = {'R06', 'R10', 'R14'}
```

---

## Repository Structure

```
physionet-mi-pipeline/
├── README.md
├── requirements.txt
├── LICENSE
├── extract.py                ← trial extraction script
├── verify.py                 ← QA / verification script
├── data/                     ← place your .edf files here
│   └── .gitkeep
├── output/                   ← extracted .h5 saved here
    └── .gitkeep
```

---


## Citation

If you use this pipeline, please cite the original dataset:

> Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. *Circulation* 101(23):e215–e220, 2000.
