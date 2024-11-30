import cv2
import numpy as np

# Check for the existence of createGuidedFilter
try:
    from cv2.ximgproc import createGuidedFilter
    HAS_XIMGPROC = True
except ImportError:
    HAS_XIMGPROC = False

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

    def custom_guided_filter(I, p, radius, eps):
        """Basic implementation of a guided filter."""
        mean_I = cv2.boxFilter(I, cv2.CV_64F, (radius, radius))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (radius, radius))
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (radius, radius))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (radius, radius))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))

        q = mean_a * I + mean_b
        return q

    # Validate input image
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a color image with 3 channels (BGR).")

    image = image.astype(np.float64) / 255
    dark_channel = get_dark_channel(image, size=15)
    atmospheric_light = get_atmospheric_light(image, dark_channel)
    transmission = get_transmission(image, atmospheric_light)
    transmission = np.clip(transmission, 0.1, 1)

    # Apply guided filter for refinement
    gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    if HAS_XIMGPROC:
        # Use OpenCV's guided filter if available
        guided_filter = createGuidedFilter(gray_image, radius=60, eps=1e-3)
        transmission_refined = guided_filter.filter(transmission.astype(np.float32))
    else:
        # Fall back to custom guided filter
        transmission_refined = custom_guided_filter(gray_image, transmission.astype(np.float32), radius=60, eps=1e-3)

    # Avoid division by zero
    transmission_refined = np.maximum(transmission_refined, 0.1)

    # Ensure transmission_refined has the correct shape
    if transmission_refined.ndim == 2:
        transmission_refined = transmission_refined[..., np.newaxis]

    # Recover the scene radiance
    result = (image - atmospheric_light) / transmission_refined + atmospheric_light
    return np.clip(result * 255, 0, 255).astype(np.uint8)