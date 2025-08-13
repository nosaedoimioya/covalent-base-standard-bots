# src/identification/MapFitterDelete.py
# This module provides the MapFitter class for fitting calibration maps.

import os
import numpy as np
from matplotlib import pyplot as plt
from typing import List
from identification.MapGenerationDelete import CalibrationMap
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
            x     = inp

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

            # Adjustments for fitting
            # y_cls is (N,) – make it (N,1) to match the sigmoid output
            y_cls = y_cls.reshape(-1, 1).astype(np.float32)

            model.fit(
                X,
                {'order': y_cls, 'modes': Y_reg},
                epochs=epochs,
                batch_size=32,
                verbose=0
            )

            # store both the scaler and the NN
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

    