import os

import cv2
from camera_capture import CameraCapture

VIDEO_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "videos"
)
os.makedirs(VIDEO_DIR, exist_ok=True)

if __name__ == "__main__":
    cap = CameraCapture(0)
    recording = False
    writer = None
    video_count = len([f for f in os.listdir(VIDEO_DIR) if f.endswith(".avi")])

    try:
        while True:
            raw_img = cap.get_picture()
            img = cap.correct_img(raw_img)
            img = cv2.flip(img, 1)
            img = cv2.resize(img, (320, 240))

            canvas = img.copy()

            if recording:
                cv2.circle(canvas, (15, 15), 8, (0, 0, 255), -1)
                cv2.putText(
                    canvas,
                    "REC",
                    (28, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                writer.write(img)

            cv2.imshow("Camera Capture", img)
            cv2.imshow("Canvas", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if not recording:
                    video_count += 1
                    path = os.path.join(VIDEO_DIR, f"video_{video_count}.avi")
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    writer = cv2.VideoWriter(path, fourcc, 30.0, (320, 240))
                    recording = True
                    print(f"Recording started: {path}")
                else:
                    recording = False
                    writer.release()
                    writer = None
                    print("Recording stopped and saved.")
            elif key == 27:
                break

    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if writer is not None:
            writer.release()
        cap.close()
        cv2.destroyAllWindows()
