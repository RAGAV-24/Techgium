import cv2
import numpy as np

def dehaze(image):
    """Apply dehazing to an input image using Dark Channel Prior."""
    def get_dark_channel(image, size):
        """Compute the dark channel of the image."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        dark_channel = cv2.erode(np.min(image, axis=2), kernel)
        return dark_channel

    def get_atmospheric_light(image, dark_channel):
        """Estimate atmospheric light."""
        flat_image = image.reshape(-1, 3)
        flat_dark = dark_channel.flatten()
        search_idx = (-flat_dark).argsort()[:int(0.001 * len(flat_dark))]
        return np.max(flat_image[search_idx], axis=0)

    def get_transmission(image, atmospheric_light, omega=0.95):
        """Estimate transmission map."""
        norm_image = image / atmospheric_light
        transmission = 1 - omega * get_dark_channel(norm_image, size=15)
        return transmission

    image = image.astype(np.float64) / 255
    dark_channel = get_dark_channel(image, size=15)
    atmospheric_light = get_atmospheric_light(image, dark_channel)
    transmission = get_transmission(image, atmospheric_light)

    transmission = np.clip(transmission, 0.1, 1)
    gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    guided_filter = cv2.ximgproc.createGuidedFilter(gray_image, 60, 1e-3)
    transmission_refined = guided_filter.filter(transmission.astype(np.float32))

    result = (image - atmospheric_light) / transmission_refined[..., None] + atmospheric_light
    return np.clip(result * 255, 0, 255).astype(np.uint8)