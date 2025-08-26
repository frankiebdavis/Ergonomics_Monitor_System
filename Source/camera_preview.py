import cv2
import sys
import platform

def get_platform_backends():
    """
    Pick capture backends per OS. Using the right backend avoids flaky camera opens.
    """
    os_name = platform.system()  # OS name: "Windows" | "Linux" | "Darwin"
    if os_name == "Windows":
        # Windows: DirectShow + Media Foundation work best
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    elif os_name == "Linux":
        # Linux: Video4Linux is the standard
        return [cv2.CAP_V4L2]
    elif os_name == "Darwin":  # macOS
        # macOS: AVFoundation is the right choice
        return [cv2.CAP_AVFOUNDATION]
    else:
        # Fallback: let OpenCV decide
        return [cv2.CAP_ANY]

def find_cameras(max_index=6, backends=None):
    """
    Probe camera indices [0..max_index-1] using given backends.
    Returns a sorted list of indices that actually open.
    """
    max_index = int(max_index)  # max index to scan (exclusive)
    backends = backends or get_platform_backends()  # list[int] backends to try

    available = []  # list[int] that opened successfully
    print(f"🔍 Scanning for cameras (indices 0 to {max_index-1}) using backends: {backends}")

    for i in range(max_index):  # i: candidate camera index
        for backend in backends:  # backend: cv2 CAP_* constant
            cap = None  # cap: cv2.VideoCapture handle
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    available.append(i)
                    print(f"  ✅ Found camera at index {i} with backend {backend}")
                    break  # first working backend is enough for this index
            except Exception as e:
                print(f"  ⚠️ Error opening camera index {i} with backend {backend}: {e}")
            finally:
                if cap is not None:
                    cap.release()

    return sorted(set(available))

def open_camera(index, backends=None):
    """
    Open a camera by index, trying multiple backends in order.
    Returns an opened cv2.VideoCapture or None.
    """
    index = int(index)  # which camera to open
    backends = backends or get_platform_backends()  # which backends to try

    for backend in backends:
        cap = None  # cap: candidate VideoCapture
        try:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                print(f"✅ Camera {index} opened successfully with backend {backend}")
                return cap
        except Exception as e:
            print(f"⚠️ Error opening camera {index} with backend {backend}: {e}")
        finally:
            # if we didn't return it, make sure it's closed
            if cap is not None and not cap.isOpened():
                cap.release()

    print(f"❌ Failed to open camera index {index} using all tested backends.")
    return None

def main():
    backends = get_platform_backends()                      # list[int] backends to try
    cameras = find_cameras(max_index=6, backends=backends)  # list[int] detected camera indices

    if not cameras:
        print("❌ No cameras detected. Check USB connection, privacy settings, and close other apps using the camera.")
        sys.exit(1)

    print(f"\n✅ Available camera indices: {cameras}")

    # pick the last index (usually the external USB cam if you have a laptop)
    selected_index = cameras[-1]  # int: chosen camera index
    print(f"📸 Trying to open camera index {selected_index}...")

    cap = open_camera(selected_index, backends)  # cv2.VideoCapture or None
    if not cap:
        sys.exit(1)

    print("🎥 Camera preview started. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()   # ret: bool frame ok; frame: np.ndarray (BGR)
            if not ret:
                print("⚠️ Frame grab failed — the webcam may be unplugged or busy.")
                break

            cv2.imshow(f"Camera {selected_index} Preview", frame)

            # quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("❎ Quitting preview.")
                break

    except KeyboardInterrupt:
        print("\n⌨️ Keyboard interrupt received. Exiting.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera released and all windows closed.")

if __name__ == "__main__":
    main()
