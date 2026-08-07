import inspect
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import dynamic_ground_perspective as perspective


def test_fixed_ground_mask_keeps_everything_at_or_below_start_row():
    mask = perspective.fixed_ground_mask(height=5, width=3, start_y=2)

    expected = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(mask, expected)


def test_metric_destination_uses_pixels_per_meter():
    destination = perspective.metric_destination(
        width_m=0.8,
        length_m=1.2,
        pixels_per_m=200.0,
    )

    expected = np.array(
        [
            [0.0, 0.0],
            [160.0, 0.0],
            [160.0, 240.0],
            [0.0, 240.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(destination, expected)


def test_homography_maps_all_four_calibration_points_to_destination():
    destination = perspective.metric_destination(
        width_m=0.8,
        length_m=1.2,
        pixels_per_m=200.0,
    )

    homography = perspective.homography_from_four_points(
        perspective.SOURCE_POINTS,
        destination,
    )

    transformed = perspective.transform_points(
        homography,
        perspective.SOURCE_POINTS,
    )
    np.testing.assert_allclose(transformed, destination, atol=1e-5)


def test_expanded_homography_translates_roi_inside_output_canvas():
    homography = np.eye(3, dtype=np.float64)
    roi_points = np.array(
        [
            [-10.0, 5.0],
            [20.0, 5.0],
            [20.0, 25.0],
            [-10.0, 25.0],
        ]
    )

    expanded, output_size = perspective.expanded_homography(
        homography,
        roi_points,
    )
    transformed = perspective.transform_points(expanded, roi_points)

    assert output_size == (31, 21)
    assert np.all(transformed >= 0.0)
    assert np.all(transformed[:, 0] <= output_size[0] - 1)
    assert np.all(transformed[:, 1] <= output_size[1] - 1)


def test_ground_roi_points_follow_mask_top_and_bottom_edges():
    mask = perspective.fixed_ground_mask(height=8, width=4, start_y=2)

    points = perspective.ground_roi_points(mask)

    expected = np.array(
        [
            [0.0, 2.0],
            [3.0, 2.0],
            [3.0, 7.0],
            [0.0, 7.0],
        ]
    )
    np.testing.assert_array_equal(points, expected)


def test_runtime_parameters_are_defined_in_module():
    assert not hasattr(perspective, "INPUT_IMAGE")
    assert perspective.CAMERA_INDEX == 0
    assert perspective.CAMERA_WIDTH == 640
    assert perspective.CAMERA_HEIGHT == 480
    assert perspective.FRAME_WIDTH == 320
    assert perspective.FRAME_HEIGHT == 240
    assert perspective.REAL_WIDTH_M > 0.0
    assert perspective.REAL_LENGTH_M > 0.0
    assert perspective.PIXELS_PER_M > 0.0
    assert perspective.GROUND_START_Y == 139
    assert not hasattr(perspective, "GroundBoundaryTracker")
    assert not hasattr(perspective, "LOWER_HSV")


def test_main_does_not_accept_command_line_parameters():
    assert list(inspect.signature(perspective.main).parameters) == []


def test_prepare_camera_frame_corrects_flips_then_resizes():
    calls = []
    raw = object()
    corrected = object()
    flipped = object()
    prepared = object()

    class CameraSpy:
        def correct_img(self, frame):
            calls.append(("correct", frame))
            return corrected

    class Cv2Spy:
        INTER_AREA = 3

        def flip(self, frame, axis):
            calls.append(("flip", frame, axis))
            return flipped

        def resize(self, frame, size, interpolation):
            calls.append(("resize", frame, size, interpolation))
            return prepared

    actual = perspective.prepare_camera_frame(raw, CameraSpy(), Cv2Spy())

    assert actual is prepared
    assert calls == [
        ("correct", raw),
        ("flip", corrected, 1),
        ("resize", flipped, (320, 240), 3),
    ]


def test_should_exit_accepts_q_and_escape_only():
    assert perspective.should_exit(ord("q"))
    assert perspective.should_exit(ord("Q"))
    assert perspective.should_exit(27)
    assert not perspective.should_exit(-1)
    assert not perspective.should_exit(ord("a"))


def test_process_camera_frame_uses_fixed_y_ground_mask(monkeypatch):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    bird_view = np.zeros((10, 20, 3), dtype=np.uint8)
    captured_masks = []

    class CameraSpy:
        def correct_img(self, raw):
            return raw

    class Cv2Spy:
        INTER_AREA = 3

        @staticmethod
        def flip(raw, _axis):
            return raw

        @staticmethod
        def resize(raw, _size, interpolation):
            assert interpolation == 3
            return raw

    def fake_warp(_frame, mask, *_args, **_kwargs):
        captured_masks.append(mask)
        return bird_view

    monkeypatch.setattr(perspective, "warp_ground", fake_warp)

    actual_frame, actual_bird, mask = perspective.process_camera_frame(
        frame,
        CameraSpy(),
        Cv2Spy(),
    )

    assert actual_frame is frame
    assert actual_bird is bird_view
    assert captured_masks == [mask]
    assert np.all(mask[:139] == 0)
    assert np.all(mask[139:] == 255)


def test_camera_stream_displays_frames_nonblocking_and_releases_resources():
    class CameraSpy:
        def __init__(self):
            self.closed = False

        def get_picture(self):
            return "raw"

        def close(self):
            self.closed = True

    class DisplaySpy:
        def __init__(self):
            self.windows = []
            self.wait_delays = []
            self.destroyed = False

        def imshow(self, name, image):
            self.windows.append((name, image))

        def waitKey(self, delay):
            self.wait_delays.append(delay)
            return 0

        def destroyAllWindows(self):
            self.destroyed = True

    frame = np.zeros((10, 15, 3), dtype=np.uint8)
    bird_view = np.zeros((20, 30, 3), dtype=np.uint8)
    ground_mask = np.zeros((10, 15), dtype=np.uint8)
    camera = CameraSpy()
    display = DisplaySpy()
    display.waitKey = lambda delay: display.wait_delays.append(delay) or ord("q")

    perspective.run_camera_stream(
        camera,
        display,
        frame_processor=lambda raw: (frame, bird_view, ground_mask),
    )

    assert camera.closed
    assert display.windows == [
        ("Camera Frame", frame),
        ("Bird's Eye View", bird_view),
        ("Ground Mask", ground_mask),
    ]
    assert display.wait_delays == [1]
    assert display.destroyed
