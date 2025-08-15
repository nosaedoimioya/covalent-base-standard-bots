# src/identification/FineTuneModelGen.py
import argparse
from typing import List

from src.identification.MapGenerationDelete import CalibrationMap
from identification.MapFitterDelete import MapFitter, ModelLoader

def _infer_dims(map_files: List[str]):
    dummy = CalibrationMap(numPositions=1, axesCommanded=1, numJoints=1)
    total_poses = 0
    num_axes = None
    num_joints = None
    for f in map_files:
        m = dummy.load_map(f)
        poses, axes = m.allWn.shape
        joints = m.initialPositions.shape[2]
        if num_axes is None:
            num_axes = axes
            num_joints = joints
        total_poses += poses
    return total_poses, num_axes, num_joints

def main():
    parser = argparse.ArgumentParser(description="Fine-tune saved shaper NN models")
    parser.add_argument("--model", required=True, help="Location file of existing models")
    parser.add_argument("maps", nargs='+', help="Calibration map pickle files for new data")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save", help="Output file for updated models")
    args = parser.parse_args()

    # Infer the dimensions based on the new maps 
    # (assumed to be same as original maps that created the model)
    num_poses, num_axes, num_joints = _infer_dims(args.maps)

    # Load the model
    loader = ModelLoader(num_axes=num_axes)
    nn_models = loader.load_models(args.model)

    # Use MapFitter for fine tuning
    fitter = MapFitter(map_names=args.maps,
                       num_positions=num_poses,
                       axes_commanded=num_axes,
                       num_joints=num_joints)
    
    fitter.fine_tune_shaper_neural_network_twohead(nn_models, lr=args.lr, epochs=args.epochs)

    loader.nn_models = fitter.nn_models
    
    # Will not overwrite existing model
    out_file = args.save if args.save else 'fine_tuned_map'
    loader.save_models(out_file)

if __name__ == "__main__":
    main()