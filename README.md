# Minimal Synkyrian Training Companion (v1)

This repository contains a minimal, self–contained implementation of the **Synkyrian Training Companion** for neural network training.

The Companion is a lightweight Keras callback that:

- Monitors the **train–validation gap** and a smoothed **validation loss slope**.
- Tracks simple “field viability” proxies  
  H_train = 1 / (1 + loss), H_val = 1 / (1 + val_loss), and their difference ΔH.
- Raises **early alarms** for emerging overfit / collapse regimes.
- Optionally applies **Synkyrian penalties** and **adaptive learning–rate control**,  
  and can trigger **early stopping** when a collapse event is detected.

The code is designed as a **didactic, minimal example** that can be easily plugged into existing Keras/TensorFlow pipelines.

---

## 1. Installation

The code assumes Python 3.10+ and TensorFlow 2.x.

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # on macOS / Linux
# .venv\Scripts\activate    # on Windows PowerShell

pip install -r requirements.txt
```

You can then run the MNIST demo directly (see below).

---

## 2. Quickstart: MNIST + Companion (online demo)

The simplest way to see the Companion in action is to run the MNIST demo script.

From the repository root:

```bash
source .venv/bin/activate             # if not already active
python train_mnist_with_companion.py
```

This will:

1. Download and preprocess the MNIST dataset.

2. Build a small convolutional classifier in Keras.

3. Attach the **SynkyrianTrainingCompanionCallback** during `model.fit`.

4. Print, at each epoch, the Synkyrian quantities:

   * train/val loss  
   * gap (T_gap = val_loss - loss)  
   * smoothed validation slope  
   * (H_train, H_val, ΔH)  
   * penalties (P_gap, P_ΔH)  
   * alarms / events / lead time

5. Evaluate the final model on the test set and print a short **Synkyrian summary**, e.g.:

```text
=== Synkyrian Companion – final summary ===
has_event: True
has_alarm: True
event_epoch: 22
alarm_epoch: 11
lead_time: 11 epochs (alarm → event)
```

This demo is intentionally simple and meant to be read as **executable documentation**.

---

## 3. Using the Companion in your own Keras model

The core interface is a single Keras callback class:

* `SynkyrianTrainingCompanionCallback`
  (defined in `synkyrian_training_companion.py`).

A minimal integration looks like this:

```python
from tensorflow import keras
from synkyrian_training_companion import SynkyrianTrainingCompanionCallback

# 1. Build your model as usual
model = keras.Sequential(
    [
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 2. Instantiate the Companion
companion_cb = SynkyrianTrainingCompanionCallback(
    verbose=1,             # 0 = silent, 1 = print Synkyrian status per epoch
    # gap_event=0.06,      # optional manual override (usually left to adaptive mode)
    # gap_alarm=0.05,      # optional manual override
    # use_adaptive_thresholds=True,
    # warmup_epochs=5,
    # adaptive_lr=True,
    # penalty_threshold=7e-4,
    # lr_reduce_factor=0.3,
    # lr_min=1e-4,
    # early_stop_on_event=True,
    # event_patience=2,
)

# 3. Train with the Companion attached
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=30,
    batch_size=128,
    callbacks=[companion_cb],
)

# 4. (Optional) inspect final state programmatically
state = companion_cb.get_state()
print("Synkyrian collapse event epoch:", state.event_epoch)
print("Synkyrian alarm epoch:", state.alarm_epoch)
```

### Adaptive thresholds

By default, the Companion:

* Runs a **warmup phase** for the first `warmup_epochs` epochs (default: 5),
* Collects statistics of the gap and ΔH,
* Sets adaptive thresholds for:

  * `gap_alarm`, `gap_event`,
  * and the penalty threshold on ΔH,

with a **cap** on `gap_event` (e.g. θ_event_max ≈ 0.08) to avoid unrealistically large event thresholds.

This allows the same code to adapt reasonably across different models and datasets, while still being conservative about what counts as a “collapse event”.

---

## 4. Offline analysis example

Besides the live callback, you can also apply the Companion logic **offline** to existing training logs.

The repository includes:

* `offline/batch_test_companion.py`
* `offline/logs/*.csv`

From the repository root:

```bash
source .venv/bin/activate
python offline/batch_test_companion.py
```

This script will:

* Load each CSV log in `offline/logs/`,
* Recompute the Synkyrian quantities per epoch,
* Detect events and alarms using simple fixed thresholds,
* Print a short classification for each run, e.g. *timely alarm*, *false alarm*, *missed event*, *no event*.

This is useful if you already have training logs and want to “retrofit” the Companion on top of them.

---

## 5. Repository contents (v1)

The key files in this minimal package are:

* `synkyrian_training_companion.py`  
  Core implementation of the **SynkyrianTrainingCompanionCallback** and the underlying state logic  
  (gap, smoothed slope, (H_train, H_val, ΔH), penalties, alarms, events).

* `train_mnist_with_companion.py`
  End–to–end MNIST demo script that wires the Companion into a small CNN and prints detailed Synkyrian logs.

* `offline/batch_test_companion.py`
  Example of how to read CSV training logs and apply the Companion **offline** to classify runs as
  *timely alarm*, *false alarm*, *missed event*, *no event*.

* `offline/logs/`
  Example MNIST logs with per–epoch metrics and Synkyrian quantities, useful for testing the offline script.

* `mnist_with_companion_log.csv` / `mnist_with_companion_controlled_log.csv` (optional, if present in the root)
  Example outputs from the online MNIST demos.

* `docs/minimal_synkyrian_training_companion.pdf` (if present)
  Technical note **“A Minimal Synkyrian Training Companion: Early Warning on Neural Network Training Runs”** describing the Companion, the underlying indices, and the MNIST experiments.

The repository is deliberately kept small and focused to make it easy to read and extend.

---

## 6. Citation / Reference

If you use the Synkyrian Training Companion in academic work, research prototypes, or internal reports, please cite both the **software release** and the **technical note**:

### Software (implementation)
Kalomoirakis, P. (2025). *Minimal Synkyrian Training Companion (v1.0)* [Software].
Zenodo. https://doi.org/10.5281/zenodo.17561539

### Technical Note (conceptual and methodological background)
Kalomoirakis, P. (2025). *A Minimal Synkyrian Training Companion: Early Warning on Neural Network Training Runs*.
Zenodo. https://doi.org/10.5281/zenodo.17561599


A simple BibTeX entry you can use:
```
@software{kalomoirakis2025_companion_software,
  author       = {Kalomoirakis, Panagiotis},
  title        = {Minimal Synkyrian Training Companion (v1.0)},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17561539},
  url          = {https://doi.org/10.5281/zenodo.17561539}
}

@misc{kalomoirakis2025_companion_note,
  author       = {Kalomoirakis, Panagiotis},
  title        = {A Minimal Synkyrian Training Companion: Early Warning on Neural Network Training Runs},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17561599},
  url          = {https://doi.org/10.5281/zenodo.17561599}
}
```
---

## 7. License

This code is released under a **research–only, non-commercial license**  
(*“Minimal Synkyrian Training Companion (v1) – Research-Only, Non-Commercial License”*).

You are free to use, modify, and redistribute the code **for research and educational purposes only**,  
subject to the conditions stated in the `LICENSE` file. Any commercial use (including use as part of a
commercial product or service) requires prior written permission from the author.


