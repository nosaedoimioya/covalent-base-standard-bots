import pathlib

from identification import SineSweepReader


def make_reader(num_poses=10, max_map_size=3):
    return SineSweepReader(
        data_folder=".",
        num_poses=num_poses,
        num_axes=1,
        robot_name="test",
        data_format="csv",
        num_joints=1,
        min_freq=0.0,
        max_freq=1.0,
        freq_space=0.1,
        max_disp=1.0,
        dwell=0.1,
        Ts=0.01,
        ctrl_config="cfg",
        max_acc=1.0,
        max_vel=1.0,
        sine_cycles=1,
        max_map_size=max_map_size,
    )


def test_compute_num_maps_rounds_up():
    reader = make_reader(num_poses=10, max_map_size=3)
    assert reader.compute_num_maps() == 4
    reader2 = make_reader(num_poses=9, max_map_size=3)
    assert reader2.compute_num_maps() == 3


def test_parse_data(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("1,2,3\n4,5,6\n")
    reader = make_reader(num_poses=1, max_map_size=1)
    parsed = reader.parse_data(str(file_path))
    assert parsed == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]