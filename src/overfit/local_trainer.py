from __future__ import absolute_import, division, print_function

import lovely_tensors as lt
from tqdm import tqdm
from datetime import datetime
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from pytorch_model_summary import summary

from ..config.conf import ResNet50_320x1024_Conf, ResNet50_192x640_Conf, ConvNeXtLarge_320x1024_Conf, Effb5_320x1024_Conf, OverfitConf
from ..models.SQLDepth import SQLdepth
from ..models.posenet.pose_cnn import PoseCNN
from ..datasets.kitti_dataset import KITTIRAWDataset, KITTIOdomDataset
from ..utils import *
from ..models.layers import *
from ..losses.loss import *
from ..data.kitti.kitti_utils.kitti_utils import *


class Overfit:
    def __init__(self, conf):
        self.conf = conf
        self.get = lambda x: conf.get(x)

        self.log_path = os.path.join(self.get('tensorboard_path'), 'overfit')
        self.log_path = os.path.join(self.log_path, self.get('model_name'))
        self.log_path = os.path.join(self.log_path, datetime.now().strftime("%Y%m%d-%H%M%S"))

        # checking height and width are multiples of 32
        assert self.get('im_sz')[0] % 32 == 0, "'height' must be a multiple of 32"
        assert self.get('im_sz')[1] % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cuda" if self.get('use_cuda') else "cpu")

        self.num_scales = len(self.get('loss_scales')) # default=[0], we only perform single scale training => num_scales=1
        self.num_input_frames = len(self.get('frame_ids_training')) # default=[0, -1, 1] => num_input_frames=3
        self.num_pose_frames = 2 if self.get('pose_model_input') == "pairs" else self.num_input_frames # default=2

        assert self.get('frame_ids_training')[0] == 0, "frame_ids_training must start with 0"

        self.use_pose_net = not (self.get('use_stereo_training') and self.get('frame_ids_training') == [0])
        # the parenthesis will be always False because frame_ids_training will never be [0] and thus
        # self.use_pose_net will be True always. I think self.get('frame_ids_training') == [0] only in supervised
        # settings and thus we basically use PoseNet when we have not stereo training and also we haven't supervised training.


        if self.get('use_stereo_training'):
            self.get('frame_ids_training').append("s")

        model = SQLdepth(conf)
        self.models["encoder"] = model.encoder
        self.models["depth_decoder"] = model.depth_decoder

        if self.get('load_pretrained_model'):
            model.from_pretrained()

        self.models["encoder"] = self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth_decoder"] = self.models["depth_decoder"].to(self.device)
        self.parameters_to_train += list(self.models["depth_decoder"].parameters())

        self.models["pose_cnn"] = PoseCNN(self.num_pose_frames) # default=2
        if self.get('load_pretrained_pose'):
            self.models["pose_cnn"].from_pretrained()

        self.models["pose_cnn"] = self.models["pose_cnn"].to(self.device)
        self.parameters_to_train += list(self.models["pose_cnn"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.get('learning_rate')) # default=1e-4

        print("Overfiting model named:\n  ", self.get('model_name'))
        print("Models and tensorboard events files are saved to:\n  ", self.get('tensorboard_path'))
        print("Overfiting is using:\n  ", self.device)
        print("Using split:\n  ", self.get('training_split'))

        # Preparing data for training
        datasets_dict = {"kitti": KITTIRAWDataset, "kitti_odom": KITTIOdomDataset}
        self.dataset = datasets_dict[self.get('training_dataset')] #default="kitti"

        fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/kitti/kitti_splits", self.get('training_split'), "{}_files.txt")

        overfit_filenames = readlines(fpath.format("overfit"))
        img_ext = '.png' if self.get('train_from_png') else '.jpg'

        num_train_samples = len(overfit_filenames)
        self.num_total_steps = num_train_samples // self.get('bs') * self.get('num_epochs') # total number of iterations

        overfit_dataset = self.dataset(self.get('data_path'), overfit_filenames, self.get('im_sz')[0], self.get('im_sz')[1],
                                     self.get('frame_ids_training'), 1, is_train=True, img_ext=img_ext) # 1 means num_scales
        self.overfit_loader = DataLoader(overfit_dataset, self.get('bs'), True, num_workers=self.get('num_workers'),
                                       pin_memory=True, drop_last=True)

        self.batch = next(iter(self.overfit_loader))

        self.writers = {}
        mode='overfit'
        self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        # Layer to compute the SSIM loss between a pair of images. We always use it.
        if self.get('use_ssim'):
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.get('loss_scales'): # default we have just 1 scale which is set to 0!
            h = self.get('im_sz')[0] // (2 ** scale) # this will do nothing basically
            w = self.get('im_sz')[1] // (2 ** scale) # also this

            # Layer to transform a depth image into a point cloud.
            self.backproject_depth[scale] = BackprojectDepth(self.get('bs'), h, w)
            self.backproject_depth[scale].to(self.device)

            # Layer which projects 3D points into a camera with intrinsics K and at position T.
            self.project_3d[scale] = Project3D(self.get('bs'), h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        self.save_opts()

    def overfit_batch(self, iters=2):
        self.step=0
        for i in (tbar := tqdm(range(iters), desc="Overfit")):
            self.model_optimizer.zero_grad()
            depth_maps_dict, depth_maps, losses = self.process_batch(self.batch)
            if "depth_gt" in self.batch:
                compute_depth_losses(self.batch, depth_maps_dict, losses, self.depth_metric_names)
            self.log("overfit", self.batch, depth_maps, depth_maps_dict, losses)
            losses["loss"].backward()
            self.model_optimizer.step()
            self.step += 1
        print("Overfit done!")

    def process_batch(self, inputs):
        """
            Pass a minibatch through the network and generate images and losses.
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.get('pose_model_type') == "shared": # default no
            # If we are using a shared encoder for both depth and pose (as advocated in monodepthv1),
            # then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.get('frame_ids_training')])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.get('batch_size')) for f in all_features]

            features = {}
            for i, k in enumerate(self.get('frame_ids_training')):
                features[k] = [f[i] for f in all_features]
            depth_maps = self.models["depth_decoder"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            depth_maps = self.models["depth_decoder"](features)

        poses=None
        if self.use_pose_net: # default=True
            poses=predict_poses(conf, self.models, inputs, features)

        depth_maps_dict = self.generate_images_pred(inputs, depth_maps, poses)
        losses = compute_losses(self.conf, inputs, depth_maps, depth_maps_dict, self.ssim)

        return depth_maps_dict, depth_maps, losses

    def generate_images_pred(self, inputs, depth_maps, poses):
        """
            Generate the warped (reprojected) color images for a minibatch.
            Generated images are saved into the 'outputs' dictionary.
        """
        depth_maps_dict={}

        for scale in self.get('loss_scales'):
            disp = depth_maps
            if self.get('monodepthv1_multiscale'):
                source_scale = scale
                depth = disp
            else:
                disp = F.interpolate(disp, [self.get('im_sz')[0], self.get('im_sz')[1]], mode="bilinear", align_corners=False)
                source_scale = 0
                depth = disp

            depth_maps_dict[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.get('frame_ids_training')[1:]):
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = poses[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175: "Learning Depth from Monocular Videos using Direct Methods"
                if self.get('pose_model_type') == "posecnn" and not self.get('use_stereo_training'):

                    axisangle = poses[("axisangle", 0, frame_id)]
                    translation = poses[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)

                depth_maps_dict[("sample", frame_id, scale)] = pix_coords

                depth_maps_dict[("color", frame_id, scale)] = F.grid_sample(inputs[("color", frame_id, source_scale)],
                                                                            depth_maps_dict[("sample", frame_id, scale)],
                                                                            padding_mode="border",
                                                                            align_corners=True)

                if not self.get('disable_automasking'):
                    depth_maps_dict[("color_identity", frame_id, scale)] = inputs[("color", frame_id, source_scale)]

        return depth_maps_dict

    def log(self, mode, inputs, depth_maps, depth_maps_dict, losses):
        """
            Write an event to the tensorboard events file.
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.get('bs'))):  # write a maxmimum of 4 images
            for s in self.get('loss_scales'):
                for frame_id in self.get('frame_ids_training'):
                    writer.add_image("input_color_image_{}_{}/{}".format(frame_id, s, j), inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image("color_pred_{}_{}/{}".format(frame_id, s, j), depth_maps_dict[("color", frame_id, s)][j].data, self.step)

                writer.add_image("predicted_disparity_{}/{}".format(s, j), normalize_image(depth_maps[j]), self.step)
                writer.add_image("automask_{}/{}".format(s, j), depth_maps_dict["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """
            Save options to tensorboard so we know what we ran this experiment with.
        """
        depth_encoder = summary(self.models["encoder"], torch.rand(1, 3, self.get('im_sz')[1], self.get('im_sz')[0]),
                                max_depth=4, show_parent_layers=True, print_summary=True)
        self.writers['overfit'].add_text('depth encoder', depth_encoder.__repr__())

        depth_decoder = summary(self.models["depth_decoder"], torch.rand(1, self.get('depth_encoder').get('model_dim'),
                                self.get('im_sz')[1] // 2, self.get('im_sz')[0] // 2), max_depth=4, show_parent_layers=True, print_summary=True)
        self.writers['overfit'].add_text('depth decoder', depth_decoder.__repr__())

        x1=torch.rand(1, 3, self.get('im_sz')[1], self.get('im_sz')[0])
        x2=torch.rand(1, 3, self.get('im_sz')[1], self.get('im_sz')[0])
        input_pose_cnn=torch.cat((x1, x2), dim=1)
        pose_cnn = summary(self.models["pose_cnn"], input_pose_cnn, max_depth=4, show_parent_layers=True, print_summary=True)
        self.writers['overfit'].add_text('pose cnn', pose_cnn.__repr__())

        self.writers['overfit'].add_text('config', self.conf.__repr__())



if __name__ == "__main__":

    lt.monkey_patch()

    conf = OverfitConf().conf  # ResNet50_320x1024_Conf, ResNet50_192x640_Conf, ConvNeXtLarge_320x1024_Conf, Effb5_320x1024_Conf

    overfit = Overfit(conf)
    overfit.overfit_batch()
