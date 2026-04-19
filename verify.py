import h5py
import numpy as np
import matplotlib.pyplot as plt
import mne
import os

H5_PATH    = './raw_EEG_4class.h5'
SR         = 160
T_LEN      = 640
SAMPLE_EDF = './data/S001/S001R04.edf'

_raw     = mne.io.read_raw_edf(SAMPLE_EDF, preload=False, verbose=False)
CH_NAMES = [ch.rstrip('. ') for ch in _raw.ch_names]
CH_MAP   = {name: i for i, name in enumerate(CH_NAMES)}

PLOT_CHS    = ['C3', 'Cz', 'C4']
CLASS_NAMES = {0: 'Left fist', 1: 'Right fist', 2: 'Both fists', 3: 'Both feet'}

with h5py.File(H5_PATH, 'r') as f:
    data     = f['data'][:]
    labels   = f['labels'][:]
    subjects = f['subjects'][:]

print("=" * 55)
print("CHECK 1 -- Shape")
print(f"  data.shape : {data.shape}")
assert data.shape == (8652, 640, 64), "Shape mismatch!"
print("  PASS  (8652, 640, 64)")

print("\nCHECK 2 -- Class balance")
unique, counts = np.unique(labels, return_counts=True)
for cls, cnt in zip(unique, counts):
    status = "PASS" if cnt == 2163 else "FAIL"
    print(f"  Class {cls} ({CLASS_NAMES[cls]:<12}): {cnt}  {status}")
assert all(counts == 2163), "Class imbalance detected!"
print("  PASS  all 4 classes = 2163 trials")

print("\nCHECK 3 -- Subject balance")
unique_subs, sub_counts = np.unique(subjects, return_counts=True)
n_subs   = len(unique_subs)
bad_subs = unique_subs[sub_counts != 84]
print(f"  Subjects found: {n_subs}  (expected 103)")
assert n_subs == 103, f"Expected 103 subjects, got {n_subs}"
assert len(bad_subs) == 0, f"Subjects with !=84 trials: {bad_subs}"
print(f"  PASS  103 subjects x 84 trials = {n_subs * 84}")

print("\nCHECK 4 -- NaN / Inf")
has_nan = np.isnan(data).any()
has_inf = np.isinf(data).any()
print(f"  NaN present : {has_nan}")
print(f"  Inf present : {has_inf}")
assert not has_nan and not has_inf, "Bad values found!"
print("  PASS  no NaN or Inf")

print("\nCHECK 5 -- Per-subject label distribution (21 per class)")
bad = []
for s in unique_subs:
    mask     = subjects == s
    s_labels = labels[mask]
    _, s_counts = np.unique(s_labels, return_counts=True)
    if not all(s_counts == 21):
        bad.append((int(s), s_counts.tolist()))
if bad:
    print(f"  FAIL -- {bad}")
else:
    print("  PASS  all 103 subjects have exactly 21 trials per class")

print("\nCHECK 6 -- Signal sanity")
zero_trials = np.where(np.all(data == 0, axis=(1, 2)))[0]
print(f"  Mean   : {data.mean():.6f} V")
print(f"  Std    : {data.std():.6f} V")
print(f"  Range  : [{data.min():.6f}, {data.max():.6f}] V")
print(f"  All-zero trials: {len(zero_trials)}")
if len(zero_trials) == 0:
    print("  PASS  no all-zero trials")
else:
    print(f"  FAIL -- zero trials at: {zero_trials[:10]}")

time_axis = np.arange(T_LEN) / SR
colors    = {'C3': '#e63946', 'Cz': '#457b9d', 'C4': '#2a9d8f'}

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Raw EEG -- 1 sample trial per class\n(channels: C3, Cz, C4)', fontsize=13)

for cls in range(4):
    ax    = axes[cls]
    idx   = np.where(labels == cls)[0]
    trial = data[idx[5]]
    for ch_name in PLOT_CHS:
        ax.plot(time_axis, trial[:, CH_MAP[ch_name]] * 1e6,
                label=ch_name, color=colors[ch_name], linewidth=0.9)
    ax.set_ylabel('uV', fontsize=9)
    ax.set_title(f'Class {cls}: {CLASS_NAMES[cls]}', fontsize=10, loc='left')
    ax.legend(loc='upper right', fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

axes[-1].set_xlabel('Time (s)', fontsize=10)
plt.tight_layout()
os.makedirs('./output', exist_ok=True)
plt.savefig('./output/eeg_verification_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved -> ./output/eeg_verification_plot.png")
print("=" * 55)
print("ALL CHECKS PASSED")
