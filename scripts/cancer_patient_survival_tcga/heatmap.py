# plot cosine similarity heatmap according to coordinates and values
def get_case_slide_dict(meta_df):
    case_slide_dict = {}

    for i in meta_df.index:
        c = meta_df.loc[i, 'case_id']
        s = meta_df.loc[i, 'slide_id']

        if c not in case_slide_dict:
            case_slide_dict[c] = [s]
        else:
            case_slide_dict[c] = case_slide_dict[c] + [s]
    return case_slide_dict


def draw_boxes(image, boxes, color='k', thickness=2):
    import cv2

    if not isinstance(boxes, list):
        boxes = [boxes]

    if not isinstance(color, list):
        color = [color] * len(boxes)

    for i, box in enumerate(boxes):
        x, y, w, h = box
        c = color[i]
        c = color_string_to_rgb(c)
        t = thickness[i] if isinstance(thickness, list) else thickness

        image = cv2.rectangle(image, (x, y), (x + w, y + h), c, t)

    return image


def extract_patch(image, box, save_path):
    import cv2

    x, y, w, h = box
    patch = image[y:y + h, x:x + w]
    cv2.imwrite(save_path, patch)


def color_string_to_rgb(color_string):
    color_string = color_string.replace(' ', '')
    if color_string.startswith('#'):
        color_string = color_string[1:]
    else:
        if color_string == 'k':
            color_string = '000000'
        elif color_string == 'r':
            color_string = 'ff0000'
        elif color_string == 'g':
            color_string = '00ff00'
        elif color_string == 'b':
            color_string = '0000ff'
        elif color_string == 'w':
            color_string = 'ffffff'
        else:
            raise ValueError(f"Unknown color string {color_string}")
    r = int(color_string[:2], 16)
    g = int(color_string[2:4], 16)
    b = int(color_string[4:], 16)
    return (r, g, b)


def get_meta(cancer_type='brca'):
    import pandas as pd

    meta_file_path = f'dataset_csv/tcga_{cancer_type.lower()}_all_clean.csv'
    meta_df = pd.read_csv(meta_file_path)
    return meta_df


def get_weights(case, cancer_type='brca'):
    import torch

    weight_path = f"input/TCGA_{cancer_type.upper()}/pt_files/weights_{case}.pt"
    weights = torch.load(weight_path)
    return weights


def get_wsi(slide):
    import os

    from openslide import OpenSlide

    wsi_path = os.path.join("../data_source/diagnostic_slides", slide)
    if not os.path.exists(wsi_path):
        print(f"WSI {wsi_path} does not exist")
        return None
    wsi = OpenSlide(wsi_path)
    return wsi


def blend_images(image1, image2, alpha=0.5):
    """
    image1: background image, numpy array (H, W, 3)
    image2: foreground image, numpy array (H, W, 3)
    alpha: blending factor
    """
    import cv2

    blended = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
    return blended


def read_coords(p_id, s_id, c='brca', work_dir="/condo/wanglab/tmhpxz9/wc-coca/WSI_tasks"):
    import os

    import torch

    embed_model = 'omiclip'

    c = c.lower()

    p = p_id + '-01'
    name = f"TCGA-{c.upper()}"

    s = s_id[:-4]

    coors_path = os.path.join(
        work_dir, name, f'20x20_image_embeddings_{embed_model}',
        f'{c}_{p}_{s}_coords.pt'
    )

    if not os.path.exists(coors_path):
        print(f"File {coors_path} does not exist")
        return None
    coords = torch.load(coors_path)
    return coords


def spatial_smoothing(heatmap, coor, patch_size=(256, 256), sigma=500):
    """
    smooth using nearest patches
    """
    import numpy as np
    import scipy.ndimage

    # create a mask
    mask = np.zeros(heatmap.shape[:2])
    for x, y in coor:
        mask[y:y + patch_size[0], x:x + patch_size[1]] = 1

    # smooth the mask
    smooth_mask = scipy.ndimage.gaussian_filter(mask, sigma=sigma)
    # save the mask
    np.save('mask.npy', mask)
    np.save('smooth_mask.npy', smooth_mask)

    # apply the mask to the heatmap
    heatmap_smooth = heatmap * smooth_mask[:, :, None]

    # fill non-mask area with white color
    # heatmap shape (H, W, 3), mask shape (H, W)
    for i in range(3):
        heatmap_smooth[:, :, i] = heatmap_smooth[:, :, i] + (1 - mask) * 255
    return heatmap_smooth


def plot_heatmap(
    coor,
    weight,
    image_path=None,
    patch_size=(256, 256),
    save_path=None,
    downsize=32,
    cmap='turbo',
    smooth=False,
    boxes=None,
    box_color='k',
    box_thickness=2,
    image_alpha=0.5
):
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from openslide import OpenSlide

    # read image
    image = OpenSlide(image_path)

    if boxes is not None:
        for box in boxes:
            x, y, w, h = box
            save_path_box = save_path.replace(
                '.jpg', f'_box_{x}_{y}_{w}_{h}.jpg'
            )
            image.read_region((x, y), 0,
                              (w, h)).convert('RGB').save(save_path_box)

    image_size = (image.dimensions[1], image.dimensions[0])
    # round up the image_size to the nearest multiple of patch_size
    image_size = (
        int(np.ceil(patch_size[0] * (image_size[0] // patch_size[0]))),
        int(np.ceil(patch_size[1] * (image_size[1] // patch_size[1])))
    )
    image_size = (image_size[0] // downsize, image_size[1] // downsize)
    patch_size = (patch_size[0] // downsize, patch_size[1] // downsize)
    coor = [(x // downsize, y // downsize) for x, y in coor]

    # convert weight to colors using a colormap
    cmap = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=weight.min(), vmax=weight.max())
    colors = cmap(norm(weight))

    heatmap = np.ones((image_size[0], image_size[1], 3)) * 255
    # use white color for background
    for i in range(len(coor)):
        x, y = coor[i]
        w = colors[i][:3] * 255
        w = w.astype(np.uint8)
        heatmap[y:y + patch_size[0], x:x + patch_size[1], :] = w

    if smooth:
        heatmap = spatial_smoothing(heatmap, coor, patch_size=patch_size, sigma=100)

    # blend heatmap with original image
    image = image.get_thumbnail((image_size[1], image_size[0]))
    image.save(save_path.replace('.jpg', '_original.jpg'))

    if boxes is not None:
        boxes = [
            (x // downsize, y // downsize, w // downsize, h // downsize)
            for x, y, w, h in boxes
        ]

        image = np.array(image)
        image_wb = draw_boxes(
            image, boxes, color=box_color, thickness=box_thickness
        )
        image_wb = cv2.cvtColor(image_wb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path.replace('.jpg', '_original_boxes.jpg'), image_wb)
    if image_alpha > 0:
        image = np.array(image)

        # pad the image to the same size as heatmap
        if image.shape[0] < heatmap.shape[0]:
            pad = heatmap.shape[0] - image.shape[0]
            image = np.pad(
                image, ((0, pad), (0, 0), (0, 0)),
                mode='constant',
                constant_values=255
            )
        if image.shape[1] < heatmap.shape[1]:
            pad = heatmap.shape[1] - image.shape[1]
            image = np.pad(
                image, ((0, 0), (0, pad), (0, 0)),
                mode='constant',
                constant_values=255
            )

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image.astype(np.uint8)
        heatmap = heatmap.astype(np.uint8)
        heatmap = blend_images(heatmap, image, alpha=image_alpha)

    # save jpeg file
    heatmap = heatmap.astype(np.uint8)
    if boxes is not None:
        heatmap = draw_boxes(
            heatmap, boxes, color=box_color, thickness=box_thickness
        )

    cv2.imwrite(save_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))


def read_boxes(file_path, slide_id):
    # read box coordinates, size, color, thickness
    import json

    with open(file_path, 'r') as f:
        boxes_configs = json.load(f)

    if slide_id not in boxes_configs:
        return None, None, None

    boxes = [
        (box['x'], box['y'], box['w'], box['h'])
        for box in boxes_configs[slide_id]
    ]
    box_colors = [box['color'] for box in boxes_configs[slide_id]]
    box_thickness = [box['thickness'] for box in boxes_configs[slide_id]]
    return boxes, box_colors, box_thickness


def main(
    case='TCGA-AN-A0AT',
    cancer_type='brca',
    downsize=32,
    image_alpha=0.5,
    cmap='jet',
    original_patch_size=(256, 256),
    smooth=False,
    output_dir='heatmaps',
    cap_max=0.99,
    boxes_config="boxes.json"
):

    import os

    import numpy as np

    meta_df = get_meta(cancer_type)
    case_slide_dict = get_case_slide_dict(meta_df)
    weights = get_weights(case, cancer_type)

    output_dir = os.path.join(output_dir, cancer_type)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    coors = {}
    sims = {}

    start = 0
    for slide_id in case_slide_dict[case]:
        coord_i = read_coords(case, slide_id, cancer_type)
        if coord_i is None:
            print(f"Skip slide {slide_id}")
            continue
        coors[slide_id] = coord_i

        end = start + len(coord_i)
        sims[slide_id] = np.array(weights[0, start:end])
        start = end

    for slide in coors:
        coor = coors[slide]
        weight = sims[slide]

        if cap_max is not None:
            # cap to quantile 0.99
            cap_max = np.quantile(weight, cap_max)
            weight = np.minimum(weight, cap_max)

        image_path = os.path.join("../data_source/diagnostic_slides", slide)
        save_path = f"{slide}_smoothed.jpg" if smooth else f"{slide}.jpg"

        os.makedirs(os.path.join(output_dir, slide), exist_ok=True)
        save_path = os.path.join(output_dir, slide, save_path)
        if boxes_config is not None:
            boxes, box_color, box_thickness = read_boxes(boxes_config, slide)
        else:
            boxes = None
            box_color = 'k'
            box_thickness = 2
        plot_heatmap(
            coor,
            weight,
            image_path=image_path,
            patch_size=original_patch_size,
            save_path=save_path,
            smooth=smooth,
            downsize=downsize,
            cmap=cmap,
            image_alpha=image_alpha,
            boxes=boxes,
            box_color=box_color,
            box_thickness=box_thickness
        )
        #print(f"Save heatmap to {save_path}")


if __name__ == '__main__':
    import fire
    fire.Fire(main)
