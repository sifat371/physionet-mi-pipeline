import mne
import numpy as np
import os
import h5py

DATA_PATH  = './data'
OUTPUT_FILE = './raw_EEG_4class.h5'
SR         = 160
T_LEN      = 640
MAX_TRIALS = 7

EXCLUDED_SUBJECTS = {38, 88, 89, 92, 100, 104}
ALL_SUBJECTS      = [s for s in range(1, 110) if s not in EXCLUDED_SUBJECTS]

TASK2_RUNS = {'R04', 'R08', 'R12'}  # T1->Left(0), T2->Right(1)
TASK4_RUNS = {'R06', 'R10', 'R14'}  # T1->Both fists(2), T2->Both feet(3)


def get_class_label(run_tag, annotation):
    if run_tag in TASK2_RUNS:
        if annotation == 'T1': return 0
        if annotation == 'T2': return 1
    elif run_tag in TASK4_RUNS:
        if annotation == 'T1': return 2
        if annotation == 'T2': return 3
    return None


def load_subject(subject):
    subject_str = f'S{subject:03d}'
    subject_dir = os.path.join(DATA_PATH, subject_str)

    eeg       = np.empty((84, T_LEN, 64), dtype=np.float32)
    labels    = np.empty((84,), dtype=np.int32)
    trial_ptr = 0

    for run_num in [4, 6, 8, 10, 12, 14]:
        fname   = f'{subject_str}R{run_num:02d}.edf'
        run_tag = f'R{run_num:02d}'
        fpath   = os.path.join(subject_dir, fname)

        raw      = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
        raw_data = raw.get_data().T
        annots   = raw.annotations

        cnt = {0: 0, 1: 0, 2: 0, 3: 0}

        for ann in annots:
            onset_idx = int(ann['onset'] * SR)
            desc      = ann['description']
            cls       = get_class_label(run_tag, desc)

            if cls is None:
                continue
            if cnt[cls] >= MAX_TRIALS:
                continue

            segment = raw_data[onset_idx: onset_idx + T_LEN, :]
            if segment.shape[0] < T_LEN:
                continue

            eeg[trial_ptr]    = segment
            labels[trial_ptr] = cls
            cnt[cls]   += 1
            trial_ptr  += 1

    assert trial_ptr == 84, f'Subject {subject}: expected 84 trials, got {trial_ptr}'
    return eeg, labels


N          = len(ALL_SUBJECTS)
all_eeg    = np.empty((N * 84, T_LEN, 64), dtype=np.float32)
all_labels = np.empty((N * 84,), dtype=np.int32)
all_subs   = np.empty((N * 84,), dtype=np.int32)

for i, subject in enumerate(ALL_SUBJECTS):
    print(f'Loading subject {subject} ({i+1}/{N}) ...')
    eeg_s, labels_s = load_subject(subject)
    start = i * 84
    all_eeg   [start: start+84] = eeg_s
    all_labels[start: start+84] = labels_s
    all_subs  [start: start+84] = subject

with h5py.File(OUTPUT_FILE, 'w') as f:
    f.create_dataset('data',     data=all_eeg,    compression='gzip')
    f.create_dataset('labels',   data=all_labels, compression='gzip')
    f.create_dataset('subjects', data=all_subs,   compression='gzip')

print(f'\nSaved -> {OUTPUT_FILE}')
print(f'Shape: {all_eeg.shape}')
