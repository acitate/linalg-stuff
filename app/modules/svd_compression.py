import numpy as np
import cv2


def compress(
    raw_image: np.ndarray, value: int, use_percentage: bool = False
) -> np.ndarray:

    if use_percentage:
        x, y, _ = raw_image.shape
        target_storage = x * y * (1 - value / 100)
        n = int(target_storage / (x + y + 1))
        n = max(1, min(n, min(x, y)))
    else:
        n = value

    R, G, B = cv2.split(raw_image)

    U_B, S_B, Vt_B = np.linalg.svd(B, full_matrices=False)
    U_R, S_R, Vt_R = np.linalg.svd(R, full_matrices=False)
    U_G, S_G, Vt_G = np.linalg.svd(G, full_matrices=False)

    R_compressed = U_R[:, :n] @ np.diag(S_R[:n]) @ Vt_R[:n, :]
    G_compressed = U_G[:, :n] @ np.diag(S_G[:n]) @ Vt_G[:n, :]
    B_compressed = U_B[:, :n] @ np.diag(S_B[:n]) @ Vt_B[:n, :]

    compressed_image = cv2.merge(
        [
            np.clip(R_compressed, 1, 255),
            np.clip(G_compressed, 1, 255),
            np.clip(B_compressed, 1, 255),
        ]
    )
    compressed_image = compressed_image.astype(np.uint8)

    return compressed_image
