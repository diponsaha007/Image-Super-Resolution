import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import calc_patch_size, convert_rgb_to_y


@calc_patch_size
def train(args):
    h5_file = h5py.File(args.output_path, "w")

    lr_patches = []
    hr_patches = []

    completed = 0

    image_paths = list(sorted(glob.glob("{}/*".format(args.images_dir))))

    np.random.shuffle(image_paths)

    if args.sample:
        image_paths = image_paths[: int(len(image_paths) * args.sample)]

    for image_path in image_paths:
        hr = pil_image.open(image_path).convert("RGB")
        hr_images = []

        if args.with_aug:
            for s in [1.0, 0.9, 0.8, 0.7, 0.6]:
                for r in [0, 90, 180, 270]:
                    tmp = hr.resize(
                        (int(hr.width * s), int(hr.height * s)),
                        resample=pil_image.BICUBIC,
                    )
                    tmp = tmp.rotate(r, expand=True)
                    hr_images.append(tmp)
        else:
            hr_images.append(hr)

        for hr in hr_images:
            hr_width = (hr.width // args.scale) * args.scale
            hr_height = (hr.height // args.scale) * args.scale
            hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
            lr = hr.resize(
                (hr.width // args.scale, hr_height // args.scale),
                resample=pil_image.BICUBIC,
            )
            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)
            hr = convert_rgb_to_y(hr)
            lr = convert_rgb_to_y(lr)

            print(f"HR Shape: {hr.shape}, LR Shape: {lr.shape}")

            for i in range(0, lr.shape[0] - args.patch_size + 1, args.scale):
                for j in range(0, lr.shape[1] - args.patch_size + 1, args.scale):
                    lr_patches.append(
                        lr[i : i + args.patch_size, j : j + args.patch_size]
                    )
                    hr_patches.append(
                        hr[
                            i * args.scale : i * args.scale
                            + args.patch_size * args.scale,
                            j * args.scale : j * args.scale
                            + args.patch_size * args.scale,
                        ]
                    )

        completed += 1
        print(f"Total Completed: {completed} Out of {len(image_paths)}")

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    print(f"LR Shape: {lr_patches.shape}, HR Shape: {hr_patches.shape}")
    h5_file.create_dataset("lr", data=lr_patches)
    h5_file.create_dataset("hr", data=hr_patches)

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, "w")

    lr_group = h5_file.create_group("lr")
    hr_group = h5_file.create_group("hr")

    completed = 0

    image_paths = list(sorted(glob.glob("{}/*".format(args.images_dir))))

    np.random.shuffle(image_paths)

    if args.sample:
        image_paths = image_paths[: int(len(image_paths) * args.sample)]

    for i, image_path in enumerate(image_paths):
        hr = pil_image.open(image_path).convert("RGB")
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize(
            (hr.width // args.scale, hr_height // args.scale),
            resample=pil_image.BICUBIC,
        )
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

        completed += 1
        print(f"Total Completed: {completed} Out of {len(image_paths)}")

    h5_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-dir",
        type=str,
        default="datasets-image\\DIV2K_valid_HR\\DIV2K_valid_HR",
    )
    parser.add_argument(
        "--output-path", type=str, default="datasets-hdf5\\DIV2K_val_x2.h5"
    )
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--sample", type=float, default=1)
    parser.add_argument("--with-aug", action="store_true")
    parser.add_argument("--eval", action="store_true", default=True)

    args = parser.parse_args()

    print(args)
    if not args.eval:
        train(args)
    else:
        eval(args)
