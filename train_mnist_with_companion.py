from __future__ import annotations

"""
Minimal MNIST training script that wires the Synkyrian Training Companion
so that anyone can plug it into their own Keras training loop.

- Uses a small CNN on MNIST.
- Attaches SynkyrianTrainingCompanionCallback to monitor overfitting / collapses.
- Prints both the live Companion stream and a compact summary at the end.
"""

import os

# Make TensorFlow a bit quieter
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf  # optional, but fine to keep
from tensorflow import keras
from tensorflow.keras import layers

from synkyrian_training_companion import SynkyrianTrainingCompanionCallback


def build_model(input_shape=(28, 28, 1), num_classes: int = 10) -> keras.Model:
    """A very standard small CNN for MNIST."""
    model = keras.Sequential(
        [
            layers.Conv2D(32, 3, activation="relu", input_shape=input_shape),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    # 1) Load MNIST data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # 2) Simple train/validation split
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    # 3) Add a channel dimension (H, W → H, W, 1)
    x_train = x_train[..., None]
    x_val = x_val[..., None]
    x_test = x_test[..., None]

    # 4) Build the model
    model = build_model()

    # 5) Instantiate the Synkyrian Training Companion
    companion_cb = SynkyrianTrainingCompanionCallback(
        verbose=1,  # set to 0 if you only want the final summary
    )

    print("=== Training with the Synkyrian Training Companion (MNIST demo) ===")

    # 6) Fit the model with the Companion attached
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=30,
        batch_size=128,
        callbacks=[companion_cb],
        verbose=1,
    )

    # 7) Evaluate on the test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[Companion demo] Final test loss={test_loss:.4f}, acc={test_acc:.4f}")

    # 8) Pull the final state from the Companion
    state = companion_cb.get_state()

    print("\n=== Synkyrian Companion – final summary ===")
    has_event = state.event_epoch is not None
    has_alarm = state.alarm_epoch is not None

    print(f"has_event: {has_event}")
    print(f"has_alarm: {has_alarm}")
    print(f"event_epoch: {state.event_epoch}")
    print(f"alarm_epoch: {state.alarm_epoch}")

    if has_event and has_alarm:
        lead_time = state.event_epoch - state.alarm_epoch
        print(f"lead_time: {lead_time} epochs (alarm → event)")
    elif has_event and not has_alarm:
        print("lead_time: N/A (collapse/overfit event with no prior alarm)")
    elif (not has_event) and has_alarm:
        print("lead_time: N/A (alarm raised but no collapse/overfit event)")
    else:
        print("lead_time: N/A (no alarm, no event)")


if __name__ == "__main__":
    main()
