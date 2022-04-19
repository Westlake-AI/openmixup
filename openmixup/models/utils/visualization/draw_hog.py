import torch
import numpy as np
try:
    from skimage import draw, exposure
except:
    print("Please install scikit-image.")


def hog_visualization(hog_feat, img_size=(224,224), orientations=9, pixels_per_cell=(8, 8)):
    """Plot Histogram of Oriented Gradients (HOG).

    hog_image : (M, N) ndarray, optional
        A visualisation of the HOG image. Only provided if `visualize` is True.
    """
    s_row, s_col = img_size
    c_row, c_col = pixels_per_cell

    n_cells_row = int(s_row // c_row)  # number of cells along row-axis
    n_cells_col = int(s_col // c_col)  # number of cells along col-axis

    if isinstance(hog_feat, torch.Tensor):
        hog_feat = hog_feat.detach().cpu().numpy()

    radius = min(c_row, c_col) // 2 - 1
    orientations_arr = np.arange(orientations)

    # set dr_arr, dc_arr to correspond to midpoints of orientation bins
    orientation_bin_midpoints = (
        np.pi * (orientations_arr + .5) / orientations)
    dr_arr = radius * np.sin(orientation_bin_midpoints)
    dc_arr = radius * np.cos(orientation_bin_midpoints)
    hog_image = np.zeros((s_row, s_col, 1), dtype=np.float32)
    for r in range(n_cells_row):
        for c in range(n_cells_col):
            for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                centre = tuple([r * c_row + c_row // 2,
                                c * c_col + c_col // 2])
                rr, cc = draw.line(int(centre[0] - dc),
                                    int(centre[1] + dr),
                                    int(centre[0] + dc),
                                    int(centre[1] - dr))
                hog_image[rr, cc, 0] += hog_feat[r, c, o]

    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10)) * 255
    hog_image = hog_image.transpose(2, 0, 1)
    
    return hog_image
