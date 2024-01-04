import torch
import numpy as np
import torch.nn.functional as F
from ..evaluation.evaluate_depth import compute_errors

def get_smooth_loss(disp, img):
    """
        Computes the smoothness loss for a disparity image.
        The color image is used for edge-aware smoothness.
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


def compute_reprojection_loss(conf, pred, target, ssim):
    """
        Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    if conf.get('use_ssim') is False:
        reprojection_loss = l1_loss
    else:
        ssim_loss = ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss

def compute_losses(conf, inputs, depth_maps, depth_maps_dict, ssim):
    """
        Compute the reprojection and smoothness losses for a minibatch.
    """
    losses = {}
    total_loss = 0

    for scale in conf.get('loss_scales'):
        loss = 0
        reprojection_losses = []

        if conf.get('monodepthv1_multiscale'):
            source_scale = scale
        else:
            source_scale = 0

        disp = depth_maps
        color = inputs[("color", 0, scale)]
        target = inputs[("color", 0, source_scale)]

        for frame_id in conf.get('frame_ids_training')[1:]:
            pred = depth_maps_dict[("color", frame_id, scale)]
            reprojection_losses.append(compute_reprojection_loss(conf, pred, target, ssim))

        reprojection_losses = torch.cat(reprojection_losses, 1)


        identity_reprojection_losses = []
        for frame_id in conf.get('frame_ids_training')[1:]:
            pred = inputs[("color", frame_id, source_scale)]
            identity_reprojection_losses.append(compute_reprojection_loss(conf, pred, target, ssim))

        identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

        if conf.get('use_average_reprojection'):
            identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
        else:
            # save both images, and do min all at once below
            identity_reprojection_loss = identity_reprojection_losses


        if conf.get('use_average_reprojection'):
            reprojection_loss = reprojection_losses.mean(1, keepdim=True)
        else:
            reprojection_loss = reprojection_losses

        if not conf.get('disable_automasking'):
            # add random numbers to break ties
            if conf.get('used_cuda') is True:
                identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
            else:
                identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape) * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
        else:
            combined = reprojection_loss

        if combined.shape[1] == 1:
            to_optimise = combined
        else:
            to_optimise, idxs = torch.min(combined, dim=1)

        if not conf.get('disable_automasking'):
            depth_maps_dict["identity_selection/{}".format(scale)] = (idxs > identity_reprojection_loss.shape[1] - 1).float()

        loss += to_optimise.mean()
        if color.shape[-2:] != disp.shape[-2:]:
            disp = F.interpolate(disp, [conf.get('im_sz')[0], conf.get('im_sz')[1]], mode="bilinear", align_corners=False)
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        # if GPU memory is not enough, you can downsample color instead
        # color = F.interpolate(color, [self.get('im_sz')[0] // 2, self.get('im_sz')[1] // 2], mode="bilinear", align_corners=False)
        smooth_loss = get_smooth_loss(norm_disp, color)
        loss += conf.get('disparity_smoothness_weight') * smooth_loss / (2 ** scale)
        total_loss += loss
        losses["loss/{}".format(scale)] = loss

    total_loss /= len(conf.get('loss_scales'))
    losses["loss"] = total_loss

    return losses

def compute_depth_losses(inputs, depth_maps_dict, losses, depth_metric_names):
    """
        Compute depth metrics, to allow monitoring during training.
        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance.
    """
    depth_pred = depth_maps_dict[("depth", 0, 0)]
    depth_pred = torch.clamp(F.interpolate(depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
    depth_pred = depth_pred.detach()

    depth_gt = inputs["depth_gt"]
    mask = depth_gt > 0

    # garg/eigen crop
    crop_mask = torch.zeros_like(mask)
    crop_mask[:, :, 153:371, 44:1197] = 1
    mask = mask * crop_mask

    depth_gt = depth_gt[mask]
    depth_pred = depth_pred[mask]
    depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

    depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

    depth_errors = compute_errors(depth_gt, depth_pred)

    for i, metric in enumerate(depth_metric_names):
        losses[metric] = np.array(depth_errors[i].cpu())
