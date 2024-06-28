import gc
import json
import os
import time
from pathlib import Path

import torch
from open_clip import create_model_from_pretrained
from PIL import Image

from encode import PRETRAINED_NETS, get_image_embeddings_large, model_paths
from wsi_utils import read_coords, read_image_patches


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='Calculate embeddings for test cases'
    )
    parser.add_argument(
        '--model', type=str, default='omiclip', help='pretrained model'
    )
    parser.add_argument(
        '--output_path', type=str, default='.', help='output path'
    )
    parser.add_argument(
        '--test_case', type=str, default='brca', help='test case'
    )
    parser.add_argument(
        '--skip_every_n', type=int, default=1, help='skip every n'
    )
    parser.add_argument(
        '--batch_size', type=int, default=500, help='batch size'
    )
    parser.add_argument('--use_gpu', action='store_true', help='use gpu')
    parser.add_argument(
        '--test',
        action='store_true',
        help='only process one sample for testing'
    )
    args = parser.parse_args()
    return args


def read_metadata(sample_metadata_path):
    with open(sample_metadata_path) as f:
        sample_metadata = json.load(f)
    return sample_metadata


def main():
    args = parse_args()

    work_dir = "/condo/wanglab/tmhpxz9/wc-coca/WSI_tasks"
    data_dir = os.path.join(work_dir, "data_source")
    coords_dir = os.path.join(data_dir, "patches", "patches")
    os.chdir(work_dir)
    print(f"Current working directory: {os.getcwd()}")

    model = args.model
    test_case = args.test_case.lower()
    skip = args.skip_every_n

    if args.use_gpu:
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    name = f"TCGA-{test_case.upper()}"
    patch_dir = os.path.join(work_dir, name, "patches")

    model_path = model_paths[model]
    output_path = args.output_path

    if output_path == ".":
        output_path = os.path.join(name, f"20x20_image_embeddings_{model}")

    PRETRAINED_NET = PRETRAINED_NETS[model]

    os.makedirs(output_path, exist_ok=True)

    if args.batch_size:
        batch_size = args.batch_size
    else:
        batch_size = 100 if DEVICE == "cpu" else 500

    model_pretrained, preprocess = create_model_from_pretrained(
        PRETRAINED_NET, device=DEVICE, pretrained=model_path
    )

    image_metadata_path = os.path.join(
        data_dir, name, f"tcga_{test_case}_wsi_paths.json"
    )
    image_metadata = read_metadata(image_metadata_path)

    if args.test:
        # randomly select one sample for testing
        import random
        sample_ids = list(image_metadata.keys())
        sample_id = random.choice(sample_ids)
        image_metadata = {sample_id: image_metadata[sample_id]}
        print(f"Testing on {sample_id}")

    for sample_id, wsi_paths in image_metadata.items():
        print(f"Processing {sample_id}")
        start = time.time()
        for wsi_path in wsi_paths:
            wsi_name = Path(wsi_path).stem

            save_path = os.path.join(
                output_path,
                f"{model}_{test_case}_{sample_id}_{wsi_name}_image_embeddings.pt"
            )

            if os.path.exists(save_path):
                print(f"Skipping {wsi_name}")
                continue

            try:
                coords = read_coords(os.path.join(coords_dir, f"{wsi_name}.h5"))
            except Exception as e:
                print(f"Error reading coordinates for {wsi_name}: {e}")
                continue

            coords = coords[::skip]

            try:
                patch_images_arr = read_image_patches(
                    os.path.join(patch_dir, f"{wsi_name}.h5")
                )
            except Exception as e:
                print(f"Error reading image patches for {wsi_name}: {e}")
                continue

            patch_images_arr = patch_images_arr[::skip]

            image_embeddings = get_image_embeddings_large(
                [Image.fromarray(arr) for arr in patch_images_arr],
                model_pretrained,
                preprocess,
                batch_size=batch_size,
                image_paths=False,
                DEVICE=DEVICE
            )
            torch.save(image_embeddings, save_path)

            # write the coordinates to a file
            coords_path = os.path.join(
                output_path, f"{test_case}_{sample_id}_{wsi_name}_coords.pt"
            )
            torch.save(coords, coords_path)

            del patch_images_arr
            del image_embeddings
            del coords

            # Call the garbage collector to ensure that the arrays are deleted immediately
            gc.collect()

        print(f"Time taken: {time.time() - start} seconds of {sample_id}")
        print(f"In total, {len(wsi_paths)} slides were processed.")


if __name__ == "__main__":
    main()
