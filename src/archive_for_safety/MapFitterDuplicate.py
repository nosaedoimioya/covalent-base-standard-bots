# src/identification/MapFitterDelete.py
# This module provides the MapFitter class for fitting calibration maps.

# DELETE THIS FILE and use the C++ binding instead.

import os
import numpy as np
from matplotlib import pyplot as plt
from typing import List
from src.archive_for_safety.MapGenerationDuplicate import CalibrationMap
from control import matlab
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, backend as K, layers, optimizers

try:
    from keras.saving import register_keras_serializable
except Exception:  # pragma: no cover - fallback for older TF
    from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="calibration")
def masked_mse(y_true, y_pred):
    """Mean squared error that ignores zero-valued targets."""
    mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)
    mse = tf.square((y_true - y_pred) * mask)
    return tf.reduce_sum(mse) / (K.epsilon() + tf.reduce_sum(mask))

class MapFitter:

    def __init__(self, map_names: List[str], num_positions: int, axes_commanded: int, num_joints: int):
        self.map = CalibrationMap(numPositions=num_positions,
                                  axesCommanded=axes_commanded,
                                  numJoints=num_joints)
        # Loader for calibration maps
        self.__loader_map = CalibrationMap(numPositions=num_positions,
                                           axesCommanded=axes_commanded,
                                           numJoints=num_joints)
        self.map_name_list = map_names

        self.num_poses = num_positions
        self.num_axes = axes_commanded

        self.load_maps()

        # Extract data from the maps
        self.v, self.r, self.wn1, \
            self.wn2, self.zeta1, self.zeta2, \
                 self.inertia, self.order = self.extract_vibration_parameters()
        
        self.nn_models : dict[str, dict[Model]] = {}
        self.models_exist = False

    def load_maps(self):
        '''
            Load the calibration maps from the specified map names.
            The maps are stored in self.map.
        '''
        start_pose = 0

        for map in self.map_name_list:
            # Load the map
            temp_map = self.__loader_map.load_map(map)

            # Get the size of the map elements
            num_poses, _ = temp_map.allWn.shape

            # Store elements in self.map
            self.map.allInertia[start_pose:start_pose+num_poses] = temp_map.allInertia
            self.map.allV[start_pose:start_pose+num_poses] = temp_map.allV
            self.map.allR[start_pose:start_pose+num_poses] = temp_map.allR
            self.map.allWn[start_pose:start_pose+num_poses] = temp_map.allWn
            self.map.allZeta[start_pose:start_pose+num_poses] = temp_map.allZeta
            self.map.initialPositions[start_pose:start_pose+num_poses] = temp_map.initialPositions
            self.map.sineSweepNumerators[start_pose:start_pose+num_poses] = temp_map.sineSweepNumerators
            self.map.sineSweepDenominators[start_pose:start_pose+num_poses] = temp_map.sineSweepDenominators

            start_pose += num_poses

    def extract_vibration_parameters(self):
        '''
            Extract vibration parameters from the loaded calibration maps.
            Returns:
                v, r, wn1, wn2, zeta1, zeta2, inertia, order
        '''
        v, r, wn1, wn2 = [[] for _ in range(self.num_axes)], [[] for _ in range(self.num_axes)], \
                        [[] for _ in range(self.num_axes)], [[] for _ in range(self.num_axes)]
        zeta1, zeta2, inertia, order = [[] for _ in range(self.num_axes)], \
                                    [[] for _ in range(self.num_axes)], [[] for _ in range(self.num_axes)], \
                                    [[] for _ in range(self.num_axes)]

        for pose in range(self.num_poses):
            for axis in range(self.num_axes):
                sys = matlab.tf(self.map.sineSweepNumerators[pose][axis],
                                self.map.sineSweepDenominators[pose][axis])
                modes = self.__find_modes(np.array(matlab.pole(sys)))

                if not modes:            # skip poses with no valid complex pair
                    continue

                # always log the primary mode
                wn1[axis].append(modes[0][0])
                zeta1[axis].append(modes[0][1])

                # second mode if available, else sentinel NaN (useful for NN masking)
                if len(modes) > 1:
                    wn2[axis].append(modes[1][0])
                    zeta2[axis].append(modes[1][1])
                    order[axis].append(2)        # 2-mode shaper
                else:
                    wn2[axis].append(np.nan)
                    zeta2[axis].append(np.nan)
                    order[axis].append(1)        # 1-mode shaper

                # ancillary data
                v[axis].append(self.map.allV[pose][axis] * 180 / np.pi)
                r[axis].append(self.map.allR[pose][axis] * 1000)
                inertia[axis].append(self.map.allInertia[pose][axis])

        return v, r, wn1, wn2, zeta1, zeta2, inertia, order
    
    def fit_shaper_neural_network_twohead(self, *, hidden=(64, 64), lr=1e-3, epochs=200):
        """
        One two-head Keras model per joint axis:
        • head 0 → binary classification (needs 1- or 2-mode shaper)
        • head 1 → regression [wn1, logζ1, wn2, logζ2]
        """
        self.nn_models = {}     # reset

        for axis in range(self.num_axes):
            X, Y_reg, y_cls, reg_w = self._prepare_axis_dataset(axis)

            # Ensure float32 for TF (fewer dtype surprises)
            X = np.asarray(X, dtype=np.float32)
            Y_reg = np.asarray(Y_reg, dtype=np.float32)
            y_cls = np.asarray(y_cls, dtype=np.float32).reshape(-1, 1)

            # --- normalization layer (replaces StandardScaler) ---
            norm = layers.Normalization(axis=-1, name="norm")
            norm.adapt(X)  # computes feature-wise mean/variance on X

            # -- architecture -------------------------------------------------
            inp   = Input(shape=(X.shape[1],), name="features")
            x = norm(inp)  # normalized features

            for units in hidden:
                x = layers.Dense(units, activation='relu')(x)

            out_cls = layers.Dense(1, activation='sigmoid', name='order')(x)
            out_reg = layers.Dense(4, activation='linear',  name='modes')(x)

            model = Model(inp, [out_cls, out_reg])
            model.compile(
                optimizer=optimizers.Adam(lr),
                loss={'order': 'binary_crossentropy', 'modes': masked_mse},
                loss_weights={'order': 1.0, 'modes': 10.0}
            )

            model.fit(
                X,
                {'order': y_cls, 'modes': Y_reg},
                epochs=epochs,
                batch_size=32,
                verbose=0
            )

            # store the NN
            self.nn_models[f'axis{axis}'] = dict(model=model)

        self.models_exist = True

    def _prepare_axis_dataset(self, axis_idx: int):
        # -- inputs ---------------------------------------------------------
        X = np.column_stack([
            self.v[axis_idx],          # deg
            self.r[axis_idx],          # mm
            self.inertia[axis_idx]     # kg · m²
        ]).astype(np.float32)

        eps = 1e-6
        # primary mode
        wn1     = np.array(self.wn1[axis_idx], dtype=np.float32)
        z1_log  = np.log(np.array(self.zeta1[axis_idx]) + eps).astype(np.float32)

        # second mode; NaN if not present
        wn2_arr    = np.array(self.wn2[axis_idx], dtype=np.float32)
        z2_log_arr = np.log(np.array(self.zeta2[axis_idx]) + eps)

        # classification label: 1 ⇔ sample has a valid second mode
        has_second = (~np.isnan(wn2_arr)).astype(np.float32)

        # replace NaNs with zeros so they act as “missing”
        wn2     = np.nan_to_num(wn2_arr, nan=0.).astype(np.float32)
        z2_log  = np.nan_to_num(z2_log_arr, nan=0.).astype(np.float32)

        # target tensor:  [wn1, z1_log, wn2, z2_log]
        Y_reg = np.column_stack([wn1, z1_log, wn2, z2_log]).astype(np.float32)

        # element-wise weights for masked-MSE
        reg_w = np.column_stack([
            np.ones_like(wn1), np.ones_like(z1_log),
            has_second,        has_second             # weight 0 when missing
        ]).astype(np.float32)

        return X, Y_reg, has_second, reg_w
    
    def fine_tune_shaper_neural_network_twohead(self, nn_models: dict, *, lr=1e-4, epochs=50):
        """Continue training preexisting two-head models on the data
        currently loaded in this ``MapFitter`` instance.

        Parameters
        ----------
        nn_models : dict
            Dictionary as returned by :func:`load_models` or
            :class:`MapLoader.load_models`.
        lr : float, optional
            Learning rate for the optimizer during fine tuning.
        epochs : int, optional
            Number of training epochs to run.
        """
        self.nn_models = nn_models
        for axis in range(self.num_axes):
            X, Y_reg, y_cls, _ = self._prepare_axis_dataset(axis)

            # Ensure float32 for TF (fewer dtype surprises)
            X = np.asarray(X, dtype=np.float32)
            Y_reg = np.asarray(Y_reg, dtype=np.float32)
            y_cls = np.asarray(y_cls, dtype=np.float32).reshape(-1, 1)

            entry = self.nn_models[f'axis{axis}']
            model = entry['model']

            # update learning rate if possible
            try:
                model.optimizer.learning_rate.assign(lr)
            except Exception:
                pass

            model.fit(
                X,
                {'order': y_cls, 'modes': Y_reg},
                epochs=epochs,
                batch_size=32,
                verbose=0,
            )

        self.nn_models[f'axis{axis}'] = dict(model=model)

        self.models_exist = True

    def visualize_data(self):
        """
        3-D scatter plots of (V, R, value) for every joint axis.

        • Sub-plot 0: ωn₁ [Hz] and ωn₂ [Hz]  
        • Sub-plot 1: ζ₁  and ζ₂

        Missing second-mode entries (stored as np.nan) are silently skipped.
        """
        for axis in range(self.num_axes):
            fig = plt.figure(figsize=(10, 8))
            fig.suptitle(f"Axis {axis} – Identified Modes", fontsize=14, weight="bold")
            cmaps = ("Reds", "Blues")   # mode-1 → Reds, mode-2 → Blues
            color_darken = 2   # darken the colors by this factor

            # ---------- natural frequency -----------------------------------
            ax1 = fig.add_subplot(2, 1, 1, projection="3d")
            for mode_idx, (wn_list, cmap) in enumerate(
                ((self.wn1[axis], cmaps[0]), (self.wn2[axis], cmaps[1]))
            ):
                wn_arr = np.asarray(wn_list, dtype=float)
                mask   = ~np.isnan(wn_arr)
                if mask.any():      # only plot existing points
                    ax1.scatter(
                        np.asarray(self.v[axis])[mask],
                        np.asarray(self.r[axis])[mask],
                        wn_arr[mask],
                        c=wn_arr[mask]*color_darken,
                        cmap=cmap,
                        edgecolors='k',
                        s=40,
                        label=f"ωn{mode_idx+1}"
                    )

            ax1.set_xlabel("V [deg]")
            ax1.set_ylabel("R [mm]")
            ax1.set_zlabel("ωn [Hz]")
            ax1.legend(loc="upper right")
            ax1.set_title("Natural frequencies")

            # ---------- damping ratio ---------------------------------------
            ax2 = fig.add_subplot(2, 1, 2, projection="3d")
            for mode_idx, (z_list, cmap) in enumerate(
                ((self.zeta1[axis], cmaps[0]), (self.zeta2[axis], cmaps[1]))
            ):
                z_arr = np.asarray(z_list, dtype=float)
                mask  = ~np.isnan(z_arr)
                if mask.any():
                    ax2.scatter(
                        np.asarray(self.v[axis])[mask],
                        np.asarray(self.r[axis])[mask],
                        z_arr[mask],
                        c=z_arr[mask]*color_darken,
                        cmap=cmap,
                        edgecolors='k',
                        s=40,
                        label=f"ζ{mode_idx+1}"
                    )

            ax2.set_xlabel("V [deg]")
            ax2.set_ylabel("R [mm]")
            ax2.set_zlabel("ζ [–]")
            ax2.legend(loc="upper right")
            ax2.set_title("Damping ratios")

            plt.tight_layout(rect=[0, 0, 1, 0.96])   # leave room for the suptitle
            plt.show()
    
    def save_models(self, directory: str):
        """
        Save all trained models to a given directory.

        Each model is saved as: <directory>/axis{i}_model.keras
        """
        os.makedirs(directory, exist_ok=True)
        print(f"Saving models to {directory}...")

        for axis, entry in self.nn_models.items():
            model_path = os.path.join(directory, f"{axis}_model.keras")

            # Save Keras model
            entry["model"].save(model_path)

    def load_models(self, directory: str) -> dict:
        """
        Load models from the given directory and populate self.nn_models.
        Returns the loaded dictionary.
        """
        print(f"Loading models from {directory}...")
        nn_models = {}

        for axis in range(self.num_axes):
            model_path = os.path.join(directory, f"axis{axis}_model.keras")

            model = keras.models.load_model(model_path, custom_objects={"masked_mse": masked_mse})

            nn_models[f"axis{axis}"] = {"model": model}

        self.nn_models = nn_models
        self.models_exist = True
        return nn_models
    
    def __find_modes(self, poles: np.ndarray,
                     real_tol: float = 1e-4,
                     zeta_min: float = 0.01) -> list[tuple[float, float]]:
        """
        Returns a list of (wn_hz, zeta) tuples sorted by ascending frequency.
        ─  Only keeps proper complex-conjugate pairs whose damping ratio ≥ zeta_min.
        """
        complex_poles = [p for p in poles if abs(np.imag(p)) > real_tol]
        modes, used = [], set()

        for i, p in enumerate(complex_poles):
            if i in used:
                continue
            # look for its conjugate
            try:
                j = next(k for k, q in enumerate(complex_poles[i + 1:], i + 1)
                        if abs(q - np.conj(p)) < real_tol)
            except StopIteration:
                continue
            used.update({i, j})
            wn = abs(p) / (2 * np.pi)          # natural frequency [Hz]
            zeta = -np.real(p) / abs(p)        # damping ratio
            if zeta >= zeta_min:
                modes.append((wn, zeta))

        return sorted(modes, key=lambda m: m[0])

class ModelLoader:
    """
    Loads / saves the per-axis two-head Keras models created by MapFitter.
    Each entry:  nn_models['axis0']  -->  {'model': keras.Model}
    """
    def __init__(self, num_axes: int):
        self.nn_models : dict[str, dict[Model]] = {}
        self.num_axes = num_axes
        self.models_exist = False

    # ---------- persistence ------------------------------------------------
    def save_models(self, directory: str):
        """
        Save all trained models to a given directory.

        Each model is saved as: <directory>/axis{i}_model.keras
        """
        os.makedirs(directory, exist_ok=True)
        print(f"Saving models to {directory}...")

        for axis, entry in self.nn_models.items():
            model_path = os.path.join(directory, f"{axis}_model.keras")

            # Save Keras model
            entry["model"].save(model_path)

    def load_models(self, directory: str) -> dict:
        """
        Load models and scalers from the given directory and populate self.nn_models.
        Returns the loaded dictionary.
        """
        print(f"Loading models from {directory}...")
        nn_models = {}

        for axis in range(self.num_axes):
            model_path = os.path.join(directory, f"axis{axis}_model.keras")

            model = keras.models.load_model(model_path, custom_objects={"masked_mse": masked_mse})

            nn_models[f"axis{axis}"] = {"model": model}

        self.nn_models = nn_models
        self.models_exist = True
        return nn_models