import cv2
import numpy as np

def dehaze(image):
    """Apply dehazing to an input image using Dark Channel Prior."""
    def get_dark_channel(image, size):
        """Compute the dark channel of the image."""
        return cv2.min(cv2.min(image[:, :, 0], image[:, :, 1]), image[:, :, 2])

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
    guided_filter = cv2.ximgproc.createGuidedFilter(cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY), 60, 1e-3)
    transmission = guided_filter.filter(transmission)

    result = (image - atmospheric_light) / transmission[..., None] + atmospheric_light
    return np.clip(result * 255, 0, 255).astype(np.uint8)
