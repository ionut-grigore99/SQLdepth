from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import torch
import lovely_tensors as lt
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from ..utils import readlines, normalize_image, disp_to_depth
from ..models.SQLDepth import SQLdepth
from ..datasets.kitti_dataset import KITTIRAWDataset
from ..config.conf import ConvNeXtLarge_320x1024_Conf, Effb5_320x1024_Conf, ResNet50_320x1024_Conf, ResNet50_192x640_Conf

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

# Models which were trained with stereo supervision were trained with a nominal baseline of 0.1 units.
# The KITTI rig has a baseline of 54cm.
# Therefore, to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4
writers = {}

current_file_path = os.path.dirname(__file__)
up_two_levels = os.path.join(current_file_path, '..')
splits_dir = os.path.join(up_two_levels, 'data', 'kitti/kitti_splits')
splits_dir = os.path.normpath(splits_dir)


def compute_errors(gt, pred):
    """
        Computation of error metrics between predicted and ground truth depths.
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """
        Apply the disparity post-processing method as introduced in Monodepthv1.
        "Unsupervised Monocular Depth Estimation With Left-Right Consistency" paper, pg 5-bottom right.
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(conf):
    """
        Evaluates a pretrained model using a specified test set.
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    get = lambda x: conf.get(x)
    writers["vis"] = SummaryWriter(os.path.join(get('tensorboard_path'), "vis"))
    pretrained_models_folder=get('pretrained_models_folder')
    disable_median_scaling=get('disable_median_scaling')
    prediction_depth_scale_factor=get('prediction_depth_scale_factor')


    assert get('evaluation_mode') in ["mono", "stereo"], "Please choose mono or stereo evaluation by setting evaluation_mode to 'mono' or 'stereo'!"

    if get('numpy_disparities_to_evaluate') is None:

        pretrained_models_folder = os.path.expanduser(pretrained_models_folder)

        assert os.path.isdir(get('pretrained_models_folder')), "Cannot find a folder at {}".format(get('pretrained_models_folder'))

        print("-> Loading weights from {}".format(get('pretrained_models_folder')))

        filenames = readlines(os.path.join(splits_dir, get('evaluation_split'), "test_files.txt"))
        encoder_path = os.path.join(get('pretrained_models_folder'), "encoder.pth")
        decoder_path = os.path.join(get('pretrained_models_folder'), "depth.pth")

        encoder_dict = torch.load(encoder_path, map_location=torch.device('cpu'))

        dataset = KITTIRAWDataset(get('data_path'), filenames, encoder_dict['height'], encoder_dict['width'], [0], 1, is_train=False)
        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=get('num_workers'), pin_memory=True, drop_last=False)

        model = SQLdepth(conf)
        model.from_pretrained() # NOTE: it's a bit weird their loading because they actually load only the weights for the
                                # decoder, for encoder they don't use torch.load(weights)!

        # I currently comment these because I will verifiy later on the server the functionality.
        # model.encoder.cuda()
        # encoder = torch.nn.DataParallel(model.encoder)
        #
        # model.depth_decoder.cuda()
        # depth_decoder = torch.nn.DataParallel(model.depth_decoder)


        pred_disps = []
        src_imgs = []
        error_maps = []

        print("-> Computing predictions with size {}x{}".format(encoder_dict['width'], encoder_dict['height']))

        step = 0
        with torch.no_grad():
            for data in dataloader:
                step = step + 1
                # input_color = data[("color", 0, 0)].cuda()
                input_color = data[("color", 0, 0)]
                breakpoint()

                if get('evaluation_post_process'): # for flipping post processing from the original Monodepth paper!
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = model(input_color)

                pred_disp = output
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if get('evaluation_post_process'): # flipping post processing from the original Monodepth paper!
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                # src_imgs.append(data[("color", 0, 0)])

        pred_disps = np.concatenate(pred_disps)
        # src_imgs = np.concatenate(src_imgs)

    else: # o prostie chestia asta oricum pentru ca nu o folosesc.
        # Load predictions from file
        print("-> Loading predictions from {}".format(get('ext_disp_to_eval')))
        pred_disps = np.load(get('numpy_disparities_to_evaluate'))

        if get('evaluate_eigen_to_benchmark'):
            eigen_to_benchmark_ids = np.load(os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))
            #IDK where to find eigen_to_benchmark_ids.npy file!.
            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if get('save_predicted_disparities'):
        output_path = os.path.join(get('pretrained_models_folder'), "disps_{}_split.npy".format(get('evaluation_split')))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)
        # src_imgs_path = os.path.join(get('pretrained_models_folder'), "src_{}_split.npy".format(get('evaluation_split')))
        # print("-> Saving src imgs to ", src_imgs_path)
        # np.save(src_imgs_path, src_imgs)

    if get('no_evaluation'):  # basically I've just calculated the predicted disparities and maybe save them.
        print("-> Evaluation disabled. Done.")
        quit()
###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
    elif get('evaluation_split') == 'benchmark':
        save_dir = os.path.join(get('load_weights_folder'), "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, get('evaluation_split'), "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")

    if get('evaluation_mode')=="stereo":
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        disable_median_scaling = True
        prediction_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = pred_disp

        if get('evaluation_split') == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        error_map = np.abs(gt_depth - pred_depth)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        error_map = np.multiply(error_map, mask)
        error_maps.append(error_map)

        pred_depth *= prediction_depth_scale_factor
        if not disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    if get('save_predicted_disparities'):
        error_map_path = os.path.join(get('pretrained_models_folder'), "error_{}_split.npy".format(get('eval_split')))
        print("-> Saving error maps to ", error_map_path)
        np.savez_compressed(error_map_path, data=np.array(error_maps, dtype="object"))
    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":

    lt.monkey_patch()

    conf = ResNet50_320x1024_Conf().conf  # ResNet50_320x1024_Conf, ResNet50_192x640_Conf, ConvNeXtLarge_320x1024_Conf, Effb5_320x1024_Conf
    get = lambda x: conf.get(x)

    evaluate(conf)


