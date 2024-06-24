import argparse

import torch
import torch.backends.cudnn as cudnn
import glob
import numpy as np
import PIL.Image as pil_image

from models import FSRCNN
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-file", type=str, default="outputs//x2//best.pth")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="datasets-image\\test_x2",
    )
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = FSRCNN(scale_factor=args.scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(
        args.weights_file, map_location=lambda storage, loc: storage
    ).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    completed = 0

    images = sorted(glob.glob("{}/*".format(args.image_dir)))

    for image_path in images:
        lr = pil_image.open(image_path).convert("RGB")
        bicubic = lr.resize(
            (lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC
        )
        bicubic.save(image_path.replace(".", "_bicubic_x{}.".format(args.scale)))

        lr, _ = preprocess(lr, device)
        _, ycbcr = preprocess(bicubic, device)

        with torch.no_grad():
            preds = model(lr).clamp(0.0, 1.0)

        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        output.save(image_path.replace(".", "_x{}.".format(args.scale)))
        completed += 1
        print("Completed: {}/{}".format(completed, len(images)))
