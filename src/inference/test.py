from __future__ import absolute_import, division, print_function

import os
import glob
import numpy as np
import lovely_tensors as lt
import PIL.Image as pil
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import torch
from torchvision import transforms


from ..models.SQLDepth import SQLdepth
from ..config.conf import OverfitConf, ResNet50_320x1024_Conf, ResNet50_192x640_Conf, ConvNeXtLarge_320x1024_Conf, Effb5_320x1024_Conf

# Models which were trained with stereo supervision were trained with a nominal baseline of 0.1 units.
# The KITTI rig has a baseline of 54cm. Therefore, to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

def test_simple(conf):
    """
        Function to predict for a single image or folder of images
    """

    if torch.cuda.is_available() and get('use_cuda'):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = SQLdepth(conf)

    feed_height = get('im_sz')[0]
    feed_width = get('im_sz')[1]
    model.to(device)
    model.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(get('image_path_inference')):
        # Only testing on a single image
        paths = [get('image_path_inference')]
        output_directory = os.path.dirname(get('image_path_inference'))
    elif os.path.isdir(get('image_path_inference')):
        # Searching folder for images
        paths = glob.glob(os.path.join(get('image_path_inference'), '*.{}'.format(get('image_extension_inference'))))
        output_directory = get('image_path_inference')
    else:
        raise Exception("Can not find get('image_path_inference'): {}".format(get('image_path_inference')))

    print("-> Predicting on {:d} test images".format(len(paths)))


    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.Resampling.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)


            # Prediction
            input_image = input_image.to(device)
            outputs = model(input_image)

            disp = outputs
            disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)

            # Saving uint16 output depth map
            to_save_dir = os.path.join(output_directory, "uint16_output_depth_maps")
            if not os.path.exists(to_save_dir):
                os.makedirs(to_save_dir)
            to_save_path = os.path.join(to_save_dir, "{}.png".format(output_name))
            to_save = (disp_resized_np * 1000).astype('uint16')
            pil.fromarray(to_save).save(to_save_path)

            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma_r') # cmap='viridis'
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}.jpeg".format(output_name))
            # plt.imsave(name_dest_im, disp_resized_np, cmap='gray') # for saving as gray depth maps
            im.save(name_dest_im) # for saving as colored depth maps

            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))
            print("                                         {}".format(name_dest_im))


    print('-> Done!')

if __name__ == '__main__':
    lt.monkey_patch()

    conf = ResNet50_320x1024_Conf().conf
    get = lambda x: conf.get(x)

    test_simple(conf)