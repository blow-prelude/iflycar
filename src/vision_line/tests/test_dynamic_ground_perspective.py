import inspect
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import dynamic_ground_perspective as perspective


def synthetic_component(
    width: int,
    height: int,
    intercept: float,
    slope: float,
) -> np.ndarray:
    component = np.zeros((height, width), dtype=bool)
    boundary = np.rint(intercept + slope * np.arange(width)).astype(int)
    for x, y in enumerate(boundary):
        component[y:, x] = True
    return component


def test_select_bottom_component_ignores_larger_floating_region():
    binary = np.zeros((60, 80), dtype=np.uint8)
    binary[2:30, 5:75] = 255
    binary[40:60, 20:60] = 255

    selected = perspective.select_bottom_connected_component(
        binary,
        min_area_ratio=0.01,
        bottom_band_height=2,
    )

    assert selected is not None
    assert not selected[10, 10]
    assert selected[50, 30]


def test_fit_top_boundary_recovers_sloped_edge():
    component = synthetic_component(
        width=80,
        height=60,
        intercept=18,
        slope=0.1,
    )

    boundary = perspective.fit_top_boundary(component, fallback_y=25)

    np.testing.assert_allclose(boundary[[0, 79]], [18, 26], atol=1)


def test_fit_top_boundary_uses_fallback_for_empty_component():
    component = np.zeros((60, 80), dtype=bool)

    boundary = perspective.fit_top_boundary(component, fallback_y=25)

    np.testing.assert_array_equal(boundary, np.full(80, 25.0))


def test_smooth_boundary_limits_frame_to_frame_motion():
    previous = np.full(8, 20.0)
    detected = np.full(8, 40.0)

    actual = perspective.smooth_boundary(
        detected,
        previous,
        alpha=1.0,
        max_step=4.0,
    )

    np.testing.assert_array_equal(actual, np.full(8, 24.0))


def test_boundary_to_mask_keeps_everything_below_each_column_boundary():
    boundary = np.array([1.0, 2.0, 3.0])

    mask = perspective.boundary_to_mask(boundary, height=5)

    expected = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [255, 255, 0],
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
    mask = perspective.boundary_to_mask(
        np.array([2.0, 3.0, 4.0, 5.0]),
        height=8,
    )

    points = perspective.ground_roi_points(mask)

    expected = np.array(
        [
            [0.0, 2.0],
            [3.0, 5.0],
            [3.0, 7.0],
            [0.0, 7.0],
        ]
    )
    np.testing.assert_array_equal(points, expected)


def test_tracker_reuses_previous_boundary_when_detection_fails():
    tracker = perspective.GroundBoundaryTracker(
        perspective.GroundDetectionConfig(
            fallback_y=25,
            min_area_ratio=0.01,
            alpha=1.0,
            max_step=50.0,
        )
    )
    component = (
        synthetic_component(
            width=80,
            height=60,
            intercept=18,
            slope=0.1,
        ).astype(np.uint8)
        * 255
    )

    _, first_boundary = tracker.update_from_blue_mask(component)
    _, second_boundary = tracker.update_from_blue_mask(np.zeros_like(component))

    np.testing.assert_array_equal(second_boundary, first_boundary)


def test_tracker_uses_fallback_without_detection_history():
    tracker = perspective.GroundBoundaryTracker(
        perspective.GroundDetectionConfig(
            fallback_y=25,
            min_area_ratio=0.01,
        )
    )

    mask, boundary = tracker.update_from_blue_mask(np.zeros((60, 80), dtype=np.uint8))

    np.testing.assert_array_equal(boundary, np.full(80, 25.0))
    assert np.all(mask[:25] == 0)
    assert np.all(mask[25:] == 255)


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
    assert perspective.FALLBACK_Y == 139


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
