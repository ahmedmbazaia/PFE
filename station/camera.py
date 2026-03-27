# =============================================================
#  Camera Module — RPi Camera capture
#  Saves sky images to data/images/ with timestamp filenames
# =============================================================

import logging
import os
from datetime import datetime

import config

logger = logging.getLogger(__name__)

_camera = None


def setup():
    """Initialize the RPi camera module."""
    global _camera

    if not config.CAMERA_ENABLED:
        logger.info("Camera disabled in config")
        return False

    # Ensure image directory exists
    os.makedirs(config.IMAGE_DIR, exist_ok=True)

    try:
        from picamera2 import Picamera2

        _camera = Picamera2()
        cam_config = _camera.create_still_configuration(
            main={"size": config.CAMERA_RESOLUTION}
        )
        _camera.configure(cam_config)
        _camera.start()
        logger.info("RPi camera initialized — resolution %s", config.CAMERA_RESOLUTION)
        return True

    except ImportError:
        logger.warning("picamera2 not available — using stub mode")
        _camera = None
        return False
    except Exception as e:
        logger.error("Camera init failed: %s", e)
        _camera = None
        return False


def capture():
    """
    Capture a single image and save to data/images/.
    Returns the full file path, or None on failure.
    """
    if _camera is None:
        return None

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sky_{timestamp}.{config.CAMERA_FORMAT}"
        filepath = os.path.join(config.IMAGE_DIR, filename)

        _camera.capture_file(filepath)
        logger.debug("Captured image: %s", filepath)
        return filepath

    except Exception as e:
        logger.error("Camera capture error: %s", e)
        return None


def shutdown():
    """Stop the camera gracefully."""
    global _camera
    if _camera is not None:
        try:
            _camera.stop()
            _camera.close()
        except Exception:
            pass
        _camera = None
        logger.info("Camera stopped")
