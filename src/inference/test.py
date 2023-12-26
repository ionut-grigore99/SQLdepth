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
    model.from_pretrained()

    model_input_height = get('im_sz')[0]
    model_input_width = get('im_sz')[1]
    model.to(device)
    model.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(get('image_path_inference')):
        # Only testing on a single image
        paths = [get('image_path_inference')]
        output_directory = os.path.join(os.path.dirname(__file__), "output")
    elif os.path.isdir(get('image_path_inference')):
        # Searching folder for images
        paths = glob.glob(os.path.join(get('image_path_inference'), '*.{}'.format(get('image_extension_inference'))))
        output_directory = os.path.join(os.path.dirname(__file__), "output")
    else:
        raise Exception("Can not find get('image_path_inference'): {}".format(get('image_path_inference')))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # Predicting on each image
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((model_input_width, model_input_height), pil.Resampling.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)


            # Prediction
            input_image = input_image.to(device)
            outputs = model(input_image)

            disp = outputs
            disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)
            breakpoint()

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma_r')  # cmap='viridis', cmap='plasma_r'
            colormapped_depth_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_depth_im)
            output_name = os.path.splitext(os.path.basename(image_path))[0]  # this is basically 'img1' or 'img2' etc.
            name_dest_im = os.path.join(output_directory, "{}_colormapped_depth_map.jpeg".format(output_name))
            im.save(name_dest_im)


            # Saving uint16 output depth map
            uint16_depth_map = (disp_resized_np * 1000).astype('uint16')
            name_dest_im = os.path.join(output_directory, "{}_uint16_depth_map.png".format(output_name))
            pil.fromarray(uint16_depth_map).save(name_dest_im)



            print("   Processed {:d} of {:d} images - saved predictions to:".format(idx + 1, len(paths)))
            print("                                         {}".format(output_directory))


    print('-> Done!')

if __name__ == '__main__':

    lt.monkey_patch()

    conf = ConvNeXtLarge_320x1024_Conf().conf # ResNet50_320x1024_Conf, ResNet50_192x640_Conf, ConvNeXtLarge_320x1024_Conf, Effb5_320x1024_Conf
    get = lambda x: conf.get(x)

    test_simple(conf)