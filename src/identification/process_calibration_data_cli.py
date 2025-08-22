"""Command-line interface for the ProcessCalibrationData extension.

This module loads the compiled :mod:`ProcessCalibrationData` pybind11
extension and exposes a small CLI that forwards its arguments to the
underlying :func:`processCalibrationData` function.
"""

from __future__ import annotations
import argparse
import importlib

try:
    from build.src.cpp.identification.ProcessCalibrationData import processCalibrationData
except Exception as exc:
    raise ImportError("ProcessCalibrationData extension not built") from exc

# def _load_extension():
#     try:  # pragma: no cover - extension may not be built
#         module = importlib.import_module("ProcessCalibrationData")
#     except Exception as exc:  # pragma: no cover - extension may not be built
#         raise ImportError("ProcessCalibrationData extension not built") from exc
#     return module.processCalibrationData


# processCalibrationData = _load_extension()

__all__ = ["processCalibrationData", "main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load sysID sine-sweep data and produce calibration maps."
    )
    parser.add_argument(
        "data_path", help="Path to data folder with sysID files"
    )
    parser.add_argument(
        "poses", type=int, help="Number of poses robot visits"
    )
    parser.add_argument(
        "axes", type=int, help="Number of commanded axes per pose"
    )
    parser.add_argument("--name", dest="robot_name", default="test", help="Robot name")
    parser.add_argument(
        "--format",
        dest="file_format",
        default="csv",
        choices=["csv", "npz", "npy"],
        help="Data file format",
    )
    parser.add_argument("--numjoints", dest="num_joints", type=int, default=6)
    parser.add_argument("--minfreq", dest="min_freq", type=float, default=1.0)
    parser.add_argument("--maxfreq", dest="max_freq", type=float, default=60.0)
    parser.add_argument("--freqspace", dest="freq_space", type=float, default=0.5)
    parser.add_argument("--mdisp", dest="max_disp", type=float, default=0.087266)
    parser.add_argument("--dwell", dest="dwell", type=float, default=0.0)
    parser.add_argument("--timestep", dest="Ts", type=float, default=0.004)
    parser.add_argument(
        "--type", dest="sysid_type", default="sine", choices=["bcb", "sine"]
    )
    parser.add_argument(
        "--ctrl", dest="ctrl_config", default="joint", choices=["joint", "task"]
    )
    parser.add_argument("--macc", dest="max_acc", type=float, default=2.0)
    parser.add_argument("--mvel", dest="max_vel", type=float, default=18.0)
    parser.add_argument("--sine-cycles", dest="sine_cycles", type=int, default=5)
    parser.add_argument(
        "--sensor", dest="sensor", default="ToolAcc", choices=["ToolAcc", "JointPos"]
    )
    parser.add_argument("--first-pose", dest="start_pose", type=int, default=0)
    parser.add_argument("--max-map-size", dest="max_map_size", type=int, default=12)
    parser.add_argument(
        "--saved-maps",
        dest="saved_maps",
        action="store_true",
        help="Load existing calibration maps instead of generating new ones",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return processCalibrationData(
        args.data_path,
        args.poses,
        args.axes,
        robot_name=args.robot_name,
        file_format=args.file_format,
        num_joints=args.num_joints,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        freq_space=args.freq_space,
        max_disp=args.max_disp,
        dwell=args.dwell,
        Ts=args.Ts,
        sysid_type=args.sysid_type,
        ctrl_config=args.ctrl_config,
        max_acc=args.max_acc,
        max_vel=args.max_vel,
        sine_cycles=args.sine_cycles,
        sensor=args.sensor,
        start_pose=args.start_pose,
        max_map_size=args.max_map_size,
        saved_maps=args.saved_maps,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
