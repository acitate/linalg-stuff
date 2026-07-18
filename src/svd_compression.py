import numpy as np
import cv2


def compress(image, n: int):
    raw_image = np.array(image)
    R, G, B = cv2.split(raw_image)

    U_B, S_B, Vt_B = np.linalg.svd(B, full_matrices=False)
    U_R, S_R, Vt_R = np.linalg.svd(R, full_matrices=False)
    U_G, S_G, Vt_G = np.linalg.svd(G, full_matrices=False)

    R_compressed = np.matrix(U_R[:, :n]) * np.diag(S_R[:n]) * np.matrix(Vt_R[:n, :])
    G_compressed = np.matrix(U_G[:, :n]) * np.diag(S_G[:n]) * np.matrix(Vt_G[:n, :])
    B_compressed = np.matrix(U_B[:, :n]) * np.diag(S_B[:n]) * np.matrix(Vt_B[:n, :])

    compressed_image = cv2.merge([np.clip(R_compressed, 1, 255), np.clip(G_compressed, 1, 255), np.clip(B_compressed, 1, 255)])
    compressed_image = compressed_image.astype(np.uint8)
    
    return compressed_image


def read_file(uploaded_file):

    if uploaded_file is not None:
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    return image


if __name__=="__main__":
    compressed = compress(r"/home/acitate/Projects/LinearAlgebra/picture.jpg", 20)
    cv2.imwrite(r"/home/acitate/Projects/LinearAlgebra/Compressed.jpg", compressed)
    