def read_coords(filepath):
    """
    Read the coordinates from the file (top left corner of the patch)
    :param filepath: path to the file
    :return: numpy array of the coordinates (x, y)
    """
    import h5py

    with h5py.File(filepath, "r") as f:
        a_group_key = list(f.keys())[0]
        ds_arr = f[a_group_key][()]  # returns as a numpy array
    return ds_arr


def read_patch(slide, coord, patch_size=(224, 224)):
    """
    Read the patch from the slide
    :param slide: slide object
    :param coords: coordinates of the top left corner of the patch
    :param patch_size: size of the patch
    :return: numpy array of the patch
    """
    import numpy as np

    # Read the patch
    patch = slide.read_region(coord, 0, patch_size)
    patch = np.array(patch)[:, :, :3]
    return patch


def get_all_patches(slide, coords, patch_size=(224, 224)):
    """
    Get all the patches from the slide
    :param slide: slide object
    :param coords: coordinates of the top left corner of the patch
    :param patch_size: size of the patch
    :return: list of patches (PIL images)
    """
    from PIL import Image

    patches = []
    for coord in coords:
        patch = read_patch(slide, coord, patch_size)
        patches.append(Image.fromarray(patch))
    return patches


def read_image_patches(patch_data_path):
    import h5py

    with h5py.File(patch_data_path, "r") as f:
        a_group_key = list(f.keys())[0]
        ds_arr = f[a_group_key][()]
    return ds_arr
