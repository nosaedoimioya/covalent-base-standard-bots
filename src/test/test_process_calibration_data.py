import pathlib
import pytest

pytest.importorskip("identification.processCalibrationData")
from identification import processCalibrationData


def test_process_calibration_data_saved_maps():
    data_dir = pathlib.Path(__file__).parent / "data"
    result = processCalibrationData(
        data_path=str(data_dir),
        poses=4,
        axes=1,
        robot_name="robot",
        file_format="npz",
        num_joints=1,
        min_freq=0.0,
        max_freq=1.0,
        freq_space=1.0,
        max_disp=1.0,
        dwell=1.0,
        Ts=1.0,
        sysid_type="sine",
        ctrl_config="joint",
        max_acc=1.0,
        max_vel=1.0,
        sine_cycles=1,
        sensor="ToolAcc",
        start_pose=0,
        max_map_size=2,
        saved_maps=True,
    )
    assert result == 2