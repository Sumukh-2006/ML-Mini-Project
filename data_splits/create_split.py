"""Dataset split generation according to approved V1 specification.

Unit of split: Complete utterance.
Split: 90% Train (11790), 5% Validation (655), 5% Test (655).
Deterministic random seed: 42.
Avoid frame leakage: Audio frames from the same utterance never cross split boundaries.
"""
import os
import json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLITS_DIR = os.path.join(ROOT, "data_splits")
ALIGN_FILE = os.path.join(ROOT, "outputs", "alignments", "full", "alignments.jsonl")
os.makedirs(SPLITS_DIR, exist_ok=True)


def create_splits():
    all_ids = []
    with open(ALIGN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            all_ids.append(rec["id"])

    assert len(all_ids) == 13100, f"Expected 13100 utterances, got {len(all_ids)}"

    # Deterministic permutation
    rng = np.random.RandomState(42)
    shuffled = rng.permutation(all_ids)

    n_total = len(shuffled)
    n_val = 655
    n_test = 655
    n_train = n_total - n_val - n_test  # 11790 (90%)

    train_ids = sorted(shuffled[:n_train])
    val_ids = sorted(shuffled[n_train:n_train + n_val])
    test_ids = sorted(shuffled[n_train + n_val:])

    assert len(train_ids) == 11790
    assert len(val_ids) == 655
    assert len(test_ids) == 655
    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(val_ids) & set(test_ids)) == 0

    for name, ids in [("train.txt", train_ids), ("val.txt", val_ids), ("test.txt", test_ids)]:
        path = os.path.join(SPLITS_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            for uid in ids:
                f.write(f"{uid}\n")
        print(f"Saved {name}: {len(ids)} utterances")

    print("Data splits successfully created and verified.")


if __name__ == "__main__":
    create_splits()
