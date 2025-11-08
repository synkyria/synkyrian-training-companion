from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
import pandas as pd


@dataclass
class SynkyrianTrainingCompanionState:
    """
    Internal state for the Synkyrian Training Companion.

    Tracks basic online quantities:

      * gap(t)      = val_loss - loss
      * smooth_val  = 3-step moving average of val_loss
      * slope(t)    = smooth_val(t) - smooth_val(t-1)
      * H_train     = 1 / (1 + loss)
      * H_val       = 1 / (1 + val_loss)
      * deltaH      = H_val - H_train

    Detects:

      * events (collapse / severe overfitting)
      * alarms (early warnings)

    Optionally adapts thresholds from a warmup window.
    """

    # --- basic thresholds (fallback) ---
    gap_event: float = 0.06
    gap_alarm: float = 0.05
    min_consec: int = 2
    min_lead: int = 2
    alarm_requires_slope: bool = True

    # --- penalty thresholds (fallback) ---
    theta_gap_penalty: float = 0.06
    theta_deltaH_penalty: float = 0.02
    lambda_gap: float = 1.0
    lambda_deltaH: float = 1.0

    # --- adaptive thresholds config ---
    use_adaptive_thresholds: bool = True
    warmup_epochs: int = 5
    theta_event_max: float = 0.08  # strict mode: do not let gap_event grow too large

    # --- internal history ---
    history_val_losses: List[float] = field(default_factory=list)
    last_smooth_val: Optional[float] = None
    consec_event_cond: int = 0
    event_epoch: Optional[int] = None  # 1-based
    alarm_epoch: Optional[int] = None  # 1-based

    # --- warmup buffers for adaptive thresholds ---
    _gaps_warmup: List[float] = field(default_factory=list)
    _deltaH_warmup: List[float] = field(default_factory=list)
    _adaptive_set: bool = False

    # --- last Synkyrian quantities (for logging/inspection) ---
    last_gap: float = 0.0
    last_slope: float = 0.0
    last_H_train: float = 0.0
    last_H_val: float = 0.0
    last_deltaH: float = 0.0
    last_P_gap: float = 0.0
    last_P_deltaH: float = 0.0

    def _maybe_set_adaptive_thresholds(self, epoch_human: int, gap: float, deltaH: float):
        """
        Collect warmup data and, at the end of warmup, set adaptive thresholds.

        We use:
          gap_event = min(mean_gap + k_event * std_gap, theta_event_max)
          gap_alarm = mean_gap + k_alarm * std_gap

        and for the deltaH penalty:
          theta_deltaH_penalty = max(0, -mean_deltaH + k_dH * std_deltaH)
        """
        if not self.use_adaptive_thresholds or self._adaptive_set:
            return

        # collect during warmup
        if epoch_human <= self.warmup_epochs:
            self._gaps_warmup.append(gap)
            self._deltaH_warmup.append(deltaH)

        # at the end of warmup → compute thresholds
        if epoch_human == self.warmup_epochs:
            gaps = np.array(self._gaps_warmup, dtype=float)
            dHs = np.array(self._deltaH_warmup, dtype=float)

            mean_gap = float(np.mean(gaps))
            std_gap = float(np.std(gaps) + 1e-8)
            mean_dH = float(np.mean(dHs))
            std_dH = float(np.std(dHs) + 1e-8)

            # slightly tight multipliers
            k_event = 2.0
            k_alarm = 1.2
            k_dH = 1.5

            # adaptive thresholds on the gap
            raw_gap_event = mean_gap + k_event * std_gap
            self.gap_event = min(raw_gap_event, self.theta_event_max)
            self.gap_alarm = mean_gap + k_alarm * std_gap

            # penalty threshold on gap (roughly at gap_event)
            self.theta_gap_penalty = self.gap_event

            # adaptive threshold for how negative deltaH we tolerate
            self.theta_deltaH_penalty = max(0.0, -mean_dH + k_dH * std_dH)

            self._adaptive_set = True

            print(
                f"[Companion][adaptive] Warmup done (epochs 1–{self.warmup_epochs}). "
                f"mean_gap={mean_gap:.4f}, std_gap={std_gap:.4f} → "
                f"gap_alarm={self.gap_alarm:.4f}, gap_event={self.gap_event:.4f}"
            )
            print(
                f"[Companion][adaptive] mean_ΔH={mean_dH:.4f}, std_ΔH={std_dH:.4f} → "
                f"θ_ΔH_penalty={self.theta_deltaH_penalty:.4f}"
            )

    def update(self, epoch: int, loss: float, val_loss: float) -> Dict[str, Any]:
        """
        Update state with new (loss, val_loss) at given epoch (0-based), compute
        Synkyrian quantities, and decide if we just hit an alarm or an event.
        """
        epoch_human = epoch + 1

        # update val_loss history
        self.history_val_losses.append(val_loss)

        # 3-step moving average for val_loss
        if len(self.history_val_losses) >= 3:
            smooth_val = float(np.mean(self.history_val_losses[-3:]))
        else:
            smooth_val = float(np.mean(self.history_val_losses))

        # slope of smoothed val_loss
        if self.last_smooth_val is None:
            slope = 0.0
        else:
            slope = smooth_val - self.last_smooth_val

        self.last_smooth_val = smooth_val

        gap = float(val_loss - loss)

        # H-train, H-val, deltaH
        H_train = 1.0 / (1.0 + float(loss))
        H_val = 1.0 / (1.0 + float(val_loss))
        deltaH = H_val - H_train

        # soft penalties (Synkyrian “pressure”)
        over_gap = max(0.0, gap - self.theta_gap_penalty)
        P_gap = self.lambda_gap * (over_gap ** 2)

        over_deltaH = max(0.0, -deltaH - self.theta_deltaH_penalty)
        P_deltaH = self.lambda_deltaH * (over_deltaH ** 2)

        # store last values
        self.last_gap = gap
        self.last_slope = slope
        self.last_H_train = H_train
        self.last_H_val = H_val
        self.last_deltaH = deltaH
        self.last_P_gap = P_gap
        self.last_P_deltaH = P_deltaH

        # 1) possibly set adaptive thresholds at end of warmup
        self._maybe_set_adaptive_thresholds(epoch_human, gap, deltaH)

        # 2) if adaptive is ON but thresholds not yet set → do not emit alarms/events
        if self.use_adaptive_thresholds and not self._adaptive_set:
            return {
                "epoch": epoch_human,
                "gap": gap,
                "slope": slope,
                "H_train": H_train,
                "H_val": H_val,
                "deltaH": deltaH,
                "P_gap": P_gap,
                "P_deltaH": P_deltaH,
                "event_epoch": self.event_epoch,
                "alarm_epoch": self.alarm_epoch,
                "has_event": False,
                "has_alarm": False,
                "just_event": False,
                "just_alarm": False,
                "lead_time": None,
                "timely_alarm": False,
            }

        # 3) main logic: event / alarm
        event_cond = (gap >= self.gap_event) and (slope >= 0.0)

        just_event = False
        just_alarm = False

        # collapse / overfitting event
        if self.event_epoch is None:
            if event_cond:
                self.consec_event_cond += 1
            else:
                self.consec_event_cond = 0

            if self.consec_event_cond >= self.min_consec:
                self.event_epoch = epoch_human
                just_event = True

        # alarm
        if self.alarm_epoch is None:
            if self.alarm_requires_slope:
                alarm_cond = (gap >= self.gap_alarm) and (slope > 0.0)
            else:
                alarm_cond = (gap >= self.gap_alarm) or (slope > 0.0)

            if alarm_cond:
                self.alarm_epoch = epoch_human
                just_alarm = True

        has_event = self.event_epoch is not None
        has_alarm = self.alarm_epoch is not None
        lead_time = None
        timely_alarm = False

        if has_event and has_alarm:
            lead_time = self.event_epoch - self.alarm_epoch
            if self.alarm_epoch < self.event_epoch and lead_time >= self.min_lead:
                timely_alarm = True

        return {
            "epoch": epoch_human,
            "gap": gap,
            "slope": slope,
            "H_train": H_train,
            "H_val": H_val,
            "deltaH": deltaH,
            "P_gap": P_gap,
            "P_deltaH": P_deltaH,
            "event_epoch": self.event_epoch,
            "alarm_epoch": self.alarm_epoch,
            "has_event": has_event,
            "has_alarm": has_alarm,
            "just_event": just_event,
            "just_alarm": just_alarm,
            "lead_time": lead_time,
            "timely_alarm": timely_alarm,
        }


class SynkyrianTrainingCompanionCallback(keras.callbacks.Callback):
    """
    Live Synkyrian Training Companion.

    - Adaptive thresholds from an initial warmup window.
    - Online detection of collapse / overfitting events and early alarms.
    - Optional adaptive learning-rate control and early stopping.
    """

    def __init__(
        self,
        # basic thresholds (fallback)
        gap_event: float = 0.06,
        gap_alarm: float = 0.05,
        min_consec: int = 2,
        min_lead: int = 2,
        alarm_requires_slope: bool = True,
        # penalty thresholds (fallback)
        theta_gap_penalty: float = 0.06,
        theta_deltaH_penalty: float = 0.02,
        lambda_gap: float = 1.0,
        lambda_deltaH: float = 1.0,
        # adaptive Synkyrian
        use_adaptive_thresholds: bool = True,
        warmup_epochs: int = 5,
        theta_event_max: float = 0.08,
        # adaptive control
        adaptive_lr: bool = True,
        penalty_threshold: float = 0.00070,  # ≈ 7e-4
        lr_reduce_factor: float = 0.3,
        lr_min: float = 1e-4,
        early_stop_on_event: bool = True,
        event_patience: int = 2,
        verbose: int = 1,
    ):
        super().__init__()
        self.state = SynkyrianTrainingCompanionState(
            gap_event=gap_event,
            gap_alarm=gap_alarm,
            min_consec=min_consec,
            min_lead=min_lead,
            alarm_requires_slope=alarm_requires_slope,
            theta_gap_penalty=theta_gap_penalty,
            theta_deltaH_penalty=theta_deltaH_penalty,
            lambda_gap=lambda_gap,
            lambda_deltaH=lambda_deltaH,
            use_adaptive_thresholds=use_adaptive_thresholds,
            warmup_epochs=warmup_epochs,
            theta_event_max=theta_event_max,
        )
        self.verbose = verbose

        # adaptive control config
        self.adaptive_lr = adaptive_lr
        self.penalty_threshold = penalty_threshold
        self.lr_reduce_factor = lr_reduce_factor
        self.lr_min = lr_min
        self.early_stop_on_event = early_stop_on_event
        self.event_patience = event_patience

        # internal adaptive state
        self._lr_reductions = 0
        self._current_lr: Optional[float] = None

    # ------------------------------------------------------------------
    # Helpers for LR handling
    # ------------------------------------------------------------------
    def _get_lr(self) -> float:
        if self._current_lr is not None:
            return self._current_lr
        opt = self.model.optimizer
        try:
            lr = float(K.get_value(opt.learning_rate))
        except AttributeError:
            try:
                lr = float(K.get_value(opt.lr))
            except AttributeError:
                lr = float(opt.learning_rate)
        self._current_lr = lr
        return lr

    def _set_lr(self, new_lr: float):
        opt = self.model.optimizer
        try:
            K.set_value(opt.learning_rate, new_lr)
        except AttributeError:
            try:
                K.set_value(opt.lr, new_lr)
            except AttributeError:
                pass
        self._current_lr = new_lr

    # ------------------------------------------------------------------
    # Keras callback hooks
    # ------------------------------------------------------------------
    def on_train_begin(self, logs=None):
        if self.verbose:
            print("=== Synkyrian Training Companion LIVE (adaptive) ===")
            mode = (
                "gap >= gap_alarm AND slope > 0"
                if self.state.alarm_requires_slope
                else "gap >= gap_alarm OR slope > 0"
            )
            print(f"Alarm rule: {mode}")
            print(
                f"Initial (fallback) gap_event={self.state.gap_event:.2f}, "
                f"gap_alarm={self.state.gap_alarm:.2f}, "
                f"θ_gap={self.state.theta_gap_penalty:.2f}, "
                f"θ_ΔH={self.state.theta_deltaH_penalty:.2f}"
            )
            if self.state.use_adaptive_thresholds:
                print(
                    f"Adaptive thresholds ON, warmup_epochs={self.state.warmup_epochs}, "
                    f"θ_event_max={self.state.theta_event_max:.2f}"
                )
            if self.adaptive_lr or self.early_stop_on_event:
                print(
                    "Adaptive control: "
                    f"lr_adapt={self.adaptive_lr}, "
                    f"penalty_threshold={self.penalty_threshold:.5f}, "
                    f"lr_reduce_factor={self.lr_reduce_factor:.1f}, "
                    f"lr_min={self.lr_min}, "
                    f"early_stop_on_event={self.early_stop_on_event}, "
                    f"event_patience={self.event_patience}"
                )
            print("-------------------------------------------")

        # initial LR snapshot
        self._current_lr = self._get_lr()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = float(logs.get("loss", 0.0))
        val_loss = float(logs.get("val_loss", loss))

        res = self.state.update(epoch, loss, val_loss)

        if self.verbose:
            msg = (
                f"[Companion] epoch {res['epoch']:3d} | "
                f"gap={res['gap']:.4f}, slope={res['slope']:.4f}, "
                f"H_train={res['H_train']:.4f}, H_val={res['H_val']:.4f}, "
                f"ΔH={res['deltaH']:.4f}, "
                f"P_gap={res['P_gap']:.5f}, P_ΔH={res['P_deltaH']:.5f}"
            )
            print(msg)

            if res["just_alarm"]:
                print(
                    f"[Companion]  → ALARM at epoch {res['alarm_epoch']} "
                    f"(gap={res['gap']:.4f}, slope={res['slope']:.4f})"
                )

            if res["just_event"]:
                print(
                    f"[Companion]  → EVENT (collapse/overfit) at epoch "
                    f"{res['event_epoch']}"
                )
                if res["has_alarm"]:
                    if res["timely_alarm"]:
                        print(
                            f"[Companion]     timely alarm: lead_time="
                            f"{res['lead_time']} epochs before the event."
                        )
                    else:
                        print(
                            f"[Companion]     alarm existed but was not timely "
                            f"(lead_time={res['lead_time']})."
                        )

        # === Adaptive control block ===
        if self.adaptive_lr:
            total_penalty = res["P_gap"] + res["P_deltaH"]
            if total_penalty >= self.penalty_threshold:
                current_lr = self._get_lr()
                if current_lr > self.lr_min:
                    new_lr = max(current_lr * self.lr_reduce_factor, self.lr_min)
                    if self.verbose:
                        print(
                            f"[Companion][control] Synkyrian penalty "
                            f"{total_penalty:.5f} >= {self.penalty_threshold:.5f} "
                            f"→ reduce LR: {current_lr:.5e} → {new_lr:.5e}"
                        )
                    self._set_lr(new_lr)
                    self._lr_reductions += 1

        # early stopping after event
        if self.early_stop_on_event and self.state.event_epoch is not None:
            epoch_human = epoch + 1
            if epoch_human >= self.state.event_epoch + self.event_patience:
                if self.verbose:
                    print(
                        "[Companion][control] Early stop: event at epoch "
                        f"{self.state.event_epoch}, current epoch={epoch_human}, "
                        f"patience={self.event_patience}."
                    )
                self.model.stop_training = True

# ----------------------------------------------------------------------
# Offline utilities for log analysis
# ----------------------------------------------------------------------

def load_log(path: str) -> pd.DataFrame:
    """
    Load a CSV training log into a pandas DataFrame.

    Expected at minimum:
      - 'epoch'
      - 'loss'
      - 'val_loss'

    Any extra columns are ignored.
    """
    return pd.read_csv(path)


def _compute_offline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns 'loss' and 'val_loss', compute:

    - gap(t)      = val_loss - loss
    - smooth_val  = 3-step moving average of val_loss
    - slope(t)    = smooth_val(t) - smooth_val(t-1)
    - H_train, H_val, deltaH

    and return a copy of the DataFrame with these extra columns.
    """
    if "epoch" in df.columns:
        df = df.sort_values("epoch").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    loss = df["loss"].to_numpy(dtype=float)
    val_loss = df["val_loss"].to_numpy(dtype=float)

    n = len(df)
    smooth_val = np.zeros(n, dtype=float)
    for t in range(n):
        start = max(0, t - 2)
        smooth_val[t] = float(np.mean(val_loss[start : t + 1]))

    slope = np.zeros(n, dtype=float)
    slope[0] = 0.0
    if n > 1:
        slope[1:] = smooth_val[1:] - smooth_val[:-1]

    gap = val_loss - loss
    H_train = 1.0 / (1.0 + loss)
    H_val = 1.0 / (1.0 + val_loss)
    deltaH = H_val - H_train

    df_out = df.copy()
    df_out["gap"] = gap
    df_out["smooth_val_loss"] = smooth_val
    df_out["slope"] = slope
    df_out["H_train"] = H_train
    df_out["H_val"] = H_val
    df_out["deltaH"] = deltaH
    return df_out


def _detect_event_epoch(
    df: pd.DataFrame,
    gap_event: float,
    min_consec: int,
) -> Optional[int]:
    """
    Detect the first collapse/overfitting event epoch (1-based).

    Event condition (strict mode):
      gap >= gap_event AND slope >= 0
    sustained for at least min_consec consecutive epochs.
    """
    consec = 0
    event_epoch: Optional[int] = None

    for idx, row in df.iterrows():
        gap = float(row["gap"])
        slope = float(row["slope"])
        cond = (gap >= gap_event) and (slope >= 0.0)

        if cond:
            consec += 1
        else:
            consec = 0

        if consec >= min_consec:
            # epochs are reported as 1-based
            event_epoch = idx + 1
            break

    return event_epoch


def _detect_alarm_epoch(
    df: pd.DataFrame,
    gap_alarm: float,
) -> Optional[int]:
    """
    Detect the first alarm epoch (1-based).

    Alarm condition (strict mode):
      gap >= gap_alarm AND slope > 0
    """
    for idx, row in df.iterrows():
        gap = float(row["gap"])
        slope = float(row["slope"])
        if (gap >= gap_alarm) and (slope > 0.0):
            return idx + 1
    return None


def analyze_log(
    df: pd.DataFrame,
    gap_event: float = 0.06,
    gap_alarm: float = 0.05,
    min_consec: int = 2,
    min_lead: int = 2,
) -> Dict[str, Any]:
    """
    Offline analysis of a training log.

    - Computes Synkyrian features from (loss, val_loss).
    - Detects event and alarm epochs using fixed thresholds.
    - Evaluates the prediction as timely_alarm / false_alarm / missed_event / no_event.
    """
    df_feats = _compute_offline_features(df)

    event_epoch = _detect_event_epoch(
        df_feats, gap_event=gap_event, min_consec=min_consec
    )
    alarm_epoch = _detect_alarm_epoch(
        df_feats, gap_alarm=gap_alarm
    )

    has_event = event_epoch is not None
    has_alarm = alarm_epoch is not None

    lead_time: Optional[int] = None
    timely_alarm = False
    classification: str

    if has_event and has_alarm:
        lead_time = event_epoch - alarm_epoch
        timely_alarm = (alarm_epoch < event_epoch) and (lead_time >= min_lead)
        if timely_alarm:
            classification = "timely_alarm"
            comment = (
                f"Timely alarm: {lead_time} epochs before event "
                f"(event at epoch {event_epoch}, alarm at {alarm_epoch})."
            )
        else:
            classification = "late_alarm"
            comment = (
                f"Alarm before event but with short lead time "
                f"(lead_time={lead_time}, event={event_epoch}, alarm={alarm_epoch})."
            )
    elif has_event and not has_alarm:
        classification = "missed_event"
        comment = (
            f"Collapse/overfit event at epoch {event_epoch} "
            f"with no prior alarm."
        )
    elif (not has_event) and has_alarm:
        classification = "false_alarm"
        comment = f"Alarm at epoch {alarm_epoch} but no collapse/overfit event."
    else:
        classification = "no_event"
        comment = "No alarm and no collapse/overfit event."

    return {
        "event_epoch": event_epoch,
        "alarm_epoch": alarm_epoch,
        "has_event": has_event,
        "has_alarm": has_alarm,
        "timely_alarm": timely_alarm,
        "lead_time": lead_time,
        "classification": classification,
        "comment": comment,
    }


    # convenience: external code (e.g. the training script) can inspect the state
    def get_state(self) -> SynkyrianTrainingCompanionState:
        return self.state
