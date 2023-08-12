import threading
import numpy
import opennsfw2
from PIL import Image
from keras import Model


PREDICTOR = None
THREAD_LOCK = threading.Lock()
MAX_PROBABILITY = 0.85


def get_predictor() -> Model:
    global PREDICTOR

    with THREAD_LOCK:
        if PREDICTOR is None:
            PREDICTOR = opennsfw2.make_open_nsfw_model()
    return PREDICTOR


def clear_predictor() -> None:
    global PREDICTOR
    PREDICTOR = None


def predict_image(target_path: str) -> bool:
    return opennsfw2.predict_image(target_path) > MAX_PROBABILITY


def predict_video(target_path: str) -> bool:
    _, probabilities = opennsfw2.predict_video_frames(video_path=target_path, frame_interval=100)
    return any(probability > MAX_PROBABILITY for probability in probabilities)
