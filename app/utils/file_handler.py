from PIL import Image
import numpy as np


def image_to_array(uploaded_image):
    image = Image.open(uploaded_image)
    image = np.array(image)

    return image


def csv_to_array(uploaded_csv):
    data = np.loadtxt(uploaded_csv, delimiter=",", skiprows=1)

    return data


def array_to_bmatrix(arr, fmt="{:.4g}"):
    """
    Convert a numpy array to a LaTeX bmatrix string.

    Parameters
    ----------
    arr : array_like
        Input array (any dimension).
    fmt : str, optional
        Format string for each element (default: '{:.4g}').

    Returns
    -------
    str
        LaTeX bmatrix string.
    """
    arr = np.asarray(arr)

    # Handle dimensions
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)  # Column vector
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)  # Flatten trailing dims

    rows = [" & ".join(fmt.format(x) for x in row) for row in arr]
    return "\\begin{bmatrix}\n" + " \\\\\n".join(rows) + "\n\\end{bmatrix}"
