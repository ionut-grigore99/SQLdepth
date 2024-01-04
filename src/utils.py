import torch


def readlines(filename):
    """
        Read all the lines in a text file and return them as a list.
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def normalize_image(im):
    """
        Rescale image pixels to span range [0, 1].
    """
    maximum = float(torch.max(im).cpu().item())
    minimum = float(torch.min(im).cpu().item())
    d = maximum - minimum if maximum != minimum else 1e5
    return (im - minimum) / d

def disp_to_depth(disp, min_depth, max_depth):
    """
        Convert network's sigmoid output into depth prediction.
        The formula for this conversion is given in the 'additional considerations' section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def sec_to_hm(t):
    """
        Convert time in seconds to time in hours, minutes and seconds.
        e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s
def sec_to_hm_str(t):
    """
        Convert time in seconds to a nice string.
        e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def rot_from_axisangle(vec):
    """
        Convert an axisangle rotation into a 4x4 transformation matrix. (adapted from https://github.com/Wallacoloo/printipi)
        Input 'vec' has to be B x 1 x 3.
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

def get_translation_matrix(translation_vector):
    """
        Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T
def transformation_from_parameters(axisangle, translation, invert=False):
    """
        Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M

def predict_poses(conf, models, inputs, features):
    """
        Predict poses between input frames for monocular sequences.
    """
    outputs = {}

    num_input_frames = len(conf.get('frame_ids_training'))
    num_pose_frames = 2 if conf.get('pose_model_input') == "pairs" else num_input_frames
    if num_pose_frames == 2:
        # In this setting, we compute the pose to each source frame via a
        # separate forward pass through the pose network.

        # select what features the pose network takes as input
        if conf.get('pose_model_type') == "shared":
            pose_feats = {f_i: features[f_i] for f_i in conf.get('frame_ids_training')}
        else:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in conf.get('frame_ids_training')}

        for f_i in conf.get('frame_ids_training')[1:]:
            if f_i != "s":
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                if conf.get('pose_model_type') == "separate_resnet":
                    pose_inputs = [models["pose_encoder"](torch.cat(pose_inputs, 1))]
                elif conf.get('pose_model_type') == "posecnn":
                    pose_inputs = torch.cat(pose_inputs, 1)

                axisangle, translation = models["pose_cnn"](pose_inputs)
                outputs[("axisangle", 0, f_i)] = axisangle # axisangle:[12, 1, 1, 3]
                outputs[("translation", 0, f_i)] = translation # translation:[12, 1, 1, 3]

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                # outputs[("cam_T_cam", 0, f_i)]: [12, 4, 4]

    else:
        # Here we input all frames to the pose net (and predict all poses) together
        if conf.get('pose_model_type') in ["separate_resnet", "posecnn"]:
            pose_inputs = torch.cat(
                [inputs[("color_aug", i, 0)] for i in conf.get('frame_ids_training') if i != "s"], 1)

            if conf.get('pose_model_type') == "separate_resnet":
                pose_inputs = [models["pose_encoder"](pose_inputs)]

        elif conf.get('pose_model_type') == "shared":
            pose_inputs = [features[i] for i in conf.get('frame_ids_training') if i != "s"]

        axisangle, translation = models["pose"](pose_inputs)

        for i, f_i in enumerate(conf.get('frame_ids_training')[1:]):
            if f_i != "s":
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, i], translation[:, i])

    return outputs


def count_parameters(model):
    total_params = 0
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params+=param
    print(f"Total Trainable Params: {total_params}")
    return total_params