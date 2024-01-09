from __future__ import absolute_import, division, print_function

import os
import glob
import numpy as np
import lovely_tensors as lt
import PIL.Image as pil
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib as mpl
import matplotlib.cm as cm
import torch
from torchvision import transforms


from ..models.SQLDepth import SQLdepth
from ..config.conf import ResNet50_320x1024_Conf, ResNet50_192x640_Conf, ConvNeXtLarge_320x1024_Conf, Effb5_320x1024_Conf

def test_simple(conf):
    """
        Function to predict depth map(s) for a single image or folder of images as input.
    """
    get = lambda x: conf.get(x)
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
        output_directory = os.path.join(os.path.dirname(__file__), "output_images")
    elif os.path.isdir(get('image_path_inference')):
        # Testing on a folder of images
        paths = glob.glob(os.path.join(get('image_path_inference'), '*.{}'.format(get('image_extension_inference'))))
        output_directory = os.path.join(os.path.dirname(__file__), "output_images")
    else:
        raise Exception("Can not find get('image_path_inference'): {}".format(get('image_path_inference')))

    print("-> Predicting on {:d} test image(s)".format(len(paths)))

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
            output_depth_map = model(input_image)

            output_depth_map_resized = torch.nn.functional.interpolate(output_depth_map, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving color mapped depth image
            output_depth_map_resized_np = output_depth_map_resized.squeeze().cpu().numpy()
            vmax = np.percentile(output_depth_map_resized_np, 95)
            # The 95th percentile is used here to ignore the top 5% of depth values which might be outliers.
            # This is a common practice to enhance the contrast of the visualization by focusing on the range where most of the data lies.
            normalizer = mpl.colors.Normalize(vmin=output_depth_map_resized_np.min(), vmax=vmax) # scale data values to the range [0, 1]. I
            mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma_r')  # choices: [cmap='viridis', cmap='plasma_r'].
            color_depth_map = (mapper.to_rgba(output_depth_map_resized_np)[:, :, :3] * 255).astype(np.uint8) # (375, 1242, 3)

            im = pil.fromarray(color_depth_map)
            output_name = os.path.splitext(os.path.basename(image_path))[0]  # this is basically 'img1' or 'img2' etc.
            output_name = os.path.join(output_directory, "{}_color_depth_map.jpeg".format(output_name))
            im.save(output_name)


            # Saving uint16 output_images depth map
            uint16_depth_map = (output_depth_map_resized_np * 1000).astype('uint16') # (375, 1242)
            output_name = os.path.splitext(os.path.basename(image_path))[0]  # this is basically 'img1' or 'img2' etc.
            output_name = os.path.join(output_directory, "{}_uint16_depth_map.png".format(output_name))
            pil.fromarray(uint16_depth_map).save(output_name)



            print("   Processed {:d} of {:d} images - saved predictions to:".format(idx + 1, len(paths)))
            print("                                         {}".format(output_directory))


    print('-> Done!')

if __name__ == '__main__':

    lt.monkey_patch()

    conf = ResNet50_320x1024_Conf().conf # ResNet50_320x1024_Conf, ResNet50_192x640_Conf, ConvNeXtLarge_320x1024_Conf, Effb5_320x1024_Conf

    test_simple(conf)
