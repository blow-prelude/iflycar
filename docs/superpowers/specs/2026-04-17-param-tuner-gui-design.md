# Parameter Tuner GUI Design

**Date:** 2026-04-17
**Author:** Claude
**Status:** Draft

## Overview

Create a PyQt5-based parameter tuning GUI for the vision line processing system. The tool provides real-time parameter adjustment with instant visual feedback.

## Requirements

- Left panel: Display area for image (320x240)
- Right panel: Parameter sliders with value display
- Real-time update when dragging sliders
- Camera integration using `CameraCapture` class
- Configurable parameter list via name/range/default lists

## Architecture

### Component Structure

```
MainWindow (QMainWindow)
├── Central Widget (QWidget)
│   └── Main Layout (QHBoxLayout)
│       ├── Left: Image Display Area (320x240)
│       │   └── QLabel with QPixmap
│       └── Right: Parameter Panel
│           └── QScrollArea
│               └── List of ParamSlider widgets
```

### Key Classes

#### 1. `ParamSlider` (QWidget)

Single parameter slider component.

**Attributes:**
- `param_name`: str - Parameter name
- `min_val`: int - Minimum value
- `max_val`: int - Maximum value
- `default_val`: int - Default value

**Signals:**
- `valueChanged(str, int)` - Emitted when slider value changes

**Components:**
- Name label (QLabel)
- Slider (QSlider)
- Value display (QLabel)

#### 2. `ParamTunerWindow` (QMainWindow)

Main application window.

**Attributes:**
- `params`: dict - Current parameter values
- `source_image`: np.ndarray - Original input image
- `process_callback`: Callable - Image processing function

**Key Methods:**
- `update_source_image(img)` - Update input image from camera
- `refresh_display()` - Re-process and display with current params
- `on_param_changed(name, value)` - Handle slider changes

### Data Flow

```
Camera Frame
    ↓
update_source_image()
    ↓
on_param_changed() [slider drag]
    ↓
process_callback(img, params)
    ↓
cv2_to_qimage()
    ↓
QLabel.setPixmap()
```

## Parameter Configuration

Parameters are defined using three lists:

```python
PARAM_NAMES = ["threshold", "dilate_iter", ...]
PARAM_RANGES = [(0, 255), (1, 10), ...]
PARAM_DEFAULTS = [180, 1, ...]
```

Sliders are dynamically generated from these lists.

## Processing Callback Interface

Users provide a callback function:

```python
def process_callback(img: np.ndarray, params: dict) -> np.ndarray:
    """
    Args:
        img: Source image (BGR format)
        params: Dictionary of current parameter values
    Returns:
        Processed image (BGR format)
    """
    # Processing logic here
    return processed_img
```

## Image Display

### OpenCV to Qt Conversion

```python
def cv2_to_qimage(img: np.ndarray) -> QImage:
    height, width = img.shape[:2]
    bytes_per_line = 3 * width

    if len(img.shape) == 2:  # Grayscale
        return QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    else:  # BGR
        return QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
```

### Display Properties

- Fixed size: 320x240
- Auto-scaling: `setScaledContents(True)`
- Format: BGR (OpenCV default) → RGB (Qt display)

## Main Function Usage

```python
def main():
    # Open camera
    cap = CameraCapture(index=0, width=320, height=240)

    # Define parameters
    PARAM_NAMES = ["threshold"]
    PARAM_RANGES = [(0, 255)]
    PARAM_DEFAULTS = [180]

    # Create tuner window
    tuner = ParamTunerWindow(
        param_names=PARAM_NAMES,
        param_ranges=PARAM_RANGES,
        param_defaults=PARAM_DEFAULTS,
        process_callback=process_callback
    )
    tuner.show()

    # Main loop
    while cap.is_opened():
        frame = cap.get_picture()
        tuner.update_source_image(frame)
        QApplication.processEvents()
```

## Test Example

```python
def test_process_callback(img: np.ndarray, params: dict) -> np.ndarray:
    """Test callback: binary thresholding"""
    threshold = params["threshold"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
```

## File Structure

```
src/vision_line/scripts/
├── tune_params.py          # Main GUI implementation
├── camera_capture.py       # Existing: Camera class
└── image_process.py        # Existing: Processing logic
```

## Dependencies

- PyQt5
- OpenCV (cv2)
- NumPy
