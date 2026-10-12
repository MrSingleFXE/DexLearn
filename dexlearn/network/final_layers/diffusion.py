import torch
from einops import rearrange, repeat

from .diffusion_util import MLPWrapper, GaussianDiffusion1D
from .mlp import BasicMLP
from dexlearn.utils.rot import proper_svd
from pytorch3d import transforms as pttf
from dexlearn.utils.RMS import Normalization


class DiffusionRTJ(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        N_out = (cfg.joint_num + 12) * cfg.traj_length
        policy_mlp_parameters = dict(
            hidden_layers_dim=[512, 256],
            output_dim=N_out,
            act="mish",
        )
        self.policy = MLPWrapper(
            channels=N_out, feature_dim=cfg.in_feat_dim, **policy_mlp_parameters
        )
        self.diffusion = GaussianDiffusion1D(self.policy, cfg.diffusion)
        self.rms = True
        if self.rms:
            self.RMS = Normalization(N_out)
        return

    def forward(self, data, global_feature):
        result_dict = {}

        # Deal with multiple grasps for one vision input
        batch_num, sample_num, pose_num, _ = data["hand_trans"].shape
        global_feature = repeat(global_feature, "b c -> (b t) c", t=sample_num)
        gt_rtj = torch.cat(
            [
                rearrange(data["hand_rot"], "b t n x y -> (b t) (n x y)"),
                rearrange(data["hand_trans"], "b t n x -> (b t) (n x)"),
                rearrange(data["hand_joint"], "b t n x -> (b t) (n x)"),
            ],
            dim=-1,
        )
        if self.rms:
            gt_rtj = self.RMS(gt_rtj)
        result_dict["loss_diffusion"] = self.diffusion(gt_rtj, global_feature)

        return result_dict

    def sample(self, global_feature, sample_num):
        global_feature = repeat(global_feature, "b c -> (b n) c", n=sample_num)
        pred_rtj, log_prob = self.diffusion.sample(cond=global_feature)
        log_prob = rearrange(log_prob, "(b t) -> b t", t=sample_num)
        if self.rms:
            pred_rtj = self.RMS.inv(pred_rtj)
        pose_num = self.cfg.traj_length
        hand_rot = rearrange(
            pred_rtj[..., : pose_num * 9],
            "(b t) (n x y) -> b t n x y",
            t=sample_num,
            n=pose_num,
            x=3,
        )
        hand_trans = rearrange(
            pred_rtj[..., pose_num * 9 : pose_num * 12],
            "(b t) (n c) -> b t n c",
            t=sample_num,
            n=pose_num,
        )

        hand_joint = rearrange(
            pred_rtj[..., pose_num * 12 :],
            "(b t) (n c) -> b t n c",
            t=sample_num,
            n=pose_num,
        )

        final_quat = pttf.matrix_to_quaternion(
            proper_svd(hand_rot.reshape(-1, 3, 3)).reshape_as(hand_rot)
        )
        robot_pose = torch.cat([hand_trans, final_quat, hand_joint], dim=-1)
        return robot_pose, log_prob


class DiffusionRT_MLPRTJ(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        N_out = 12
        policy_mlp_parameters = dict(
            hidden_layers_dim=[512, 256],
            output_dim=N_out,
            act="mish",
        )
        self.policy = MLPWrapper(
            channels=N_out, feature_dim=cfg.in_feat_dim, **policy_mlp_parameters
        )
        self.diffusion = GaussianDiffusion1D(self.policy, cfg.diffusion)
        self.rms = True
        if self.rms:
            self.RMS = Normalization(N_out)
        self.joint_mlp = BasicMLP(
            cfg.in_feat_dim + 12,
            (cfg.joint_num + 12) * cfg.traj_length - 12,
        )
        self.joint_loss = torch.nn.SmoothL1Loss(reduction="none")
        return

    def forward(self, data, global_feature):
        result_dict = {}

        # Deal with multiple grasps for one vision input
        batch_num, sample_num, pose_num, _ = data["hand_trans"].shape
        hand_trans = rearrange(data["hand_trans"], "b t n x -> (b t) n x")
        hand_rot = rearrange(data["hand_rot"], "b t n x y -> (b t) n x y")
        hand_joint = rearrange(data["hand_joint"], "b t n x -> (b t) n x")
        global_feature = repeat(global_feature, "b c -> (b t) c", t=sample_num)

        # Use Flow to predict grasp rot and trans
        grasp_rt = torch.cat(
            [repeat(hand_rot[:, -1], "b x y -> b (x y)"), hand_trans[:, -1]], dim=-1
        )
        if self.rms:
            grasp_rt_diff = self.RMS(grasp_rt)
        else:
            grasp_rt_diff = grasp_rt
        result_dict["loss_diffusion"] = self.diffusion(grasp_rt_diff, global_feature)

        # use MLP to predict other things
        in_mlp_feat = torch.cat([global_feature, grasp_rt], dim=-1)
        out_info = self.joint_mlp(in_mlp_feat)
        gt_info = torch.cat(
            [
                rearrange(hand_trans[:, :-1], "b n x -> b (n x)"),
                rearrange(hand_rot[:, :-1], "b n x y -> b (n x y)"),
                rearrange(hand_joint, "b n x -> b (n x)"),
            ],
            dim=-1,
        )
        loss_others = self.joint_loss(out_info, gt_info)
        pose_num = self.cfg.traj_length - 1
        result_dict["loss_trans"] = loss_others[:, : pose_num * 3].mean()
        result_dict["loss_rot"] = loss_others[:, pose_num * 3 : pose_num * 12].mean()
        result_dict["loss_joint"] = loss_others[:, pose_num * 12 :].mean()
        with torch.no_grad():
            result_dict["abs_dis_joint"] = (
                (out_info - gt_info)[..., pose_num * 12 :].abs().mean()
            )

        return result_dict

    def sample(self, global_feature, sample_num):
        global_feature = repeat(global_feature, "b c -> (b n) c", n=sample_num)
        grasp_rt, log_prob = self.diffusion.sample(cond=global_feature)
        log_prob = rearrange(log_prob, "(b t) -> b t", t=sample_num)
        if self.rms:
            grasp_rt = self.RMS.inv(grasp_rt)
        grasp_rot = proper_svd(rearrange(grasp_rt[..., :9], "b (x y) -> b x y", x=3))
        grasp_trans = grasp_rt[..., 9:]
        in_mlp_feat = torch.cat(
            [
                global_feature,
                rearrange(grasp_rot, "b x y -> b (x y)"),
                grasp_trans,
            ],
            dim=-1,
        )
        pred_info = self.joint_mlp(in_mlp_feat)
        pose_num = self.cfg.traj_length - 1
        hand_trans = rearrange(
            pred_info[:, : pose_num * 3],
            "(b t) (n c) -> b t n c",
            t=sample_num,
            n=pose_num,
        )
        hand_rot = rearrange(
            pred_info[:, pose_num * 3 : pose_num * 12],
            "(b t) (n x y) -> b t n x y",
            t=sample_num,
            n=pose_num,
            x=3,
        )

        hand_joint = rearrange(
            pred_info[:, pose_num * 12 :],
            "(b t) (n c) -> b t n c",
            t=sample_num,
            n=pose_num + 1,
        )

        final_trans = torch.cat(
            [hand_trans, rearrange(grasp_trans, "(b n) c -> b n 1 c", n=sample_num)],
            dim=-2,
        )
        final_quat = pttf.matrix_to_quaternion(
            torch.cat(
                [
                    proper_svd(hand_rot.reshape(-1, 3, 3)).reshape_as(hand_rot),
                    rearrange(grasp_rot, "(b n) x y -> b n 1 x y", n=sample_num),
                ],
                dim=-3,
            ),
        )
        robot_pose = torch.cat([final_trans, final_quat, hand_joint], dim=-1)
        return robot_pose, log_prob
