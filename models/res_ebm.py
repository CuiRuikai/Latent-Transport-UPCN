import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from timm.models.layers import trunc_normal_

from models.model_utils import PointNet_SA_Module_KNN, MLP_Res, Transformer, attention_unit, mlp, mlp_conv, Block
from extensions.chamfer_dist import ChamferDistanceL2, PatialChamferDistanceL2
from .build import MODELS


class FeatureExtractor(nn.Module):
    def __init__(self, out_dim=1024):
        """Encoder that encodes information of partial point cloud
        """
        super(FeatureExtractor, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(128, dim=64)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = Transformer(256, dim=64)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

    def forward(self, point_cloud):
        """
        Args:
             point_cloud: b, 3, n

        Returns:
            l3_points: (B, out_dim, 1)
        """
        point_cloud = point_cloud.permute(0, 2, 1).contiguous()
        l0_xyz = point_cloud
        l0_points = point_cloud

        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l1_points = self.transformer_1(l1_points, l1_xyz)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        l2_points = self.transformer_2(l2_points, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)

        return l3_points


class FeatDiscriminator(nn.Module):
    def __init__(self, feat_dim=1024):
        super(FeatDiscriminator, self).__init__()
        self.model = nn. Sequential(
           spectral_norm(nn.Conv1d(feat_dim, 1024, 1)),
           nn.LeakyReLU(0.2, inplace=True),
           spectral_norm(nn.Conv1d(1024, 512, 1)),
           nn.LeakyReLU(0.2, inplace=True),
           spectral_norm(nn.Conv1d(512, 256, 1)),
           nn.LeakyReLU(0.2, inplace=True),
           spectral_norm(nn.Conv1d(256, 64, 1)),
           nn.LeakyReLU(0.2, inplace=True),
           spectral_norm(nn.Conv1d(64, 8, 1)),
           nn.LeakyReLU(0.2, inplace=True),
           spectral_norm(nn.Conv1d(8, 1, 1))
        )

    def forward(self, x):
        x = self.model(x).squeeze(-1)
        return x


class PointDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        start_number=32
        self.in_channels = 3
        self.start_number = start_number
        self.mlp_conv1 = mlp_conv(in_channels=self.in_channels, layer_dim=[self.start_number, self.start_number * 2])
        self.attention_unit = attention_unit(in_channels=self.start_number*4)
        self.mlp_conv2 = mlp_conv(
            in_channels=self.start_number*4, layer_dim=[self.start_number*4, self.start_number*8])
        self.mlp = mlp(in_channels=self.start_number*8, layer_dim=[self.start_number * 8, 1])

    def forward(self, inputs):
        inputs = inputs.transpose(2, 1).contiguous()
        features = self.mlp_conv1(inputs)
        features_global = torch.max(features, dim=2)[0]  # global feature
        features = torch.cat([features, features_global.unsqueeze(2).repeat(1, 1, features.shape[2])], dim=1)
        features = self.attention_unit(features)

        features = self.mlp_conv2(features)
        features = torch.max(features, dim=2)[0]
        features = self.mlp(features)
        return features


class AttentionDecoder(nn.Module):
    def __init__(self, latent_dim=1024, num_output=2048, embed_dim=4, depth=1, num_heads=1, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        in_chans = 1
        self.num_features = self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_output = num_output

        self.input_proj = nn.Sequential(
            nn.Conv1d(in_chans, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(embed_dim, embed_dim, 1)
        )

        self.encoder = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth)])

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * embed_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 3 * self.num_output)
        )

    def forward(self, z):
        bs = z.size(0)
        z = z.transpose(1,2)
        x = self.input_proj(z).transpose(1,2)

        # decoder
        for i, blk in enumerate(self.encoder):
            x = blk(x)
            # x = blk(x + pos)

        x = x.reshape(bs, -1)

        point_cloud = self.mlp(x).reshape(bs, -1, 3)  #  B M C(3)

        return point_cloud


@MODELS.register_module()
class ResEBMUPCN(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.build_modules(config)
        self.build_loss_func(config)

    def build_modules(self, config):
        self.encoder = FeatureExtractor(out_dim=config.latent_dim)
        self.feat_trans = EBM(dim=config.latent_dim, step_size=config.step_size, n_step=config.n_step, noise_scale=config.noise_scale)
        self.decoder = AttentionDecoder(latent_dim=config.latent_dim, num_output=config.num_pc, embed_dim=config.embed_dim, num_heads=config.num_heads, depth=config.depth)
        self.point_disc = PointDiscriminator()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def build_loss_func(self, config):
        self.recon_criterion = ChamferDistanceL2()
        self.fidelity_criterion = PatialChamferDistanceL2()
        self.recon_weight = config.recon_weight
        self.fidelity_weight = config.fidelity_weight
        self.feat_adv_loss = config.feat_adv_loss
        self.point_adv_weight = config.point_adv_weight
        self.alpha = config.alpha

    def get_disc_loss(self, x_hat, y_hat, x_latent, y_latent):
        loss_feat = torch.zeros(1).to(x_hat)
        loss_point = F.relu(1.0 - self.point_disc(y_hat.detach())).mean() + F.relu(1.0 + self.point_disc(x_hat.detach())).mean()
        loss_all = loss_feat + loss_point

        return loss_all, loss_feat, loss_point

    def get_gen_loss(self, x, y, x_hat, y_hat, x_latent, y_latent):
        loss_recon = self.recon_criterion(y_hat, y).mean()
        loss_fidelity = self.fidelity_criterion(x, x_hat).mean()
        loss_adv_point = - self.point_disc(x_hat).mean()
        loss_all = self.recon_weight * loss_recon + self.fidelity_weight * loss_fidelity + self.point_adv_weight * loss_adv_point
        return loss_all, loss_recon, loss_fidelity, loss_adv_point

    def get_ebm_loss(self, x, y):
        with torch.no_grad():
            z = self.encoder(x)
            x_latent = z + self.feat_trans.sample_langevin(z)
            y_latent = self.encoder(y)
        real_out = self.feat_trans.energy(y_latent.detach())  # maybe I need to prevent gradient flowing to the encoder
        fake_out = self.feat_trans.energy(x_latent.detach())
        reg_loss = self.alpha * (real_out ** 2 + fake_out ** 2).mean()
        cdiv_loss = real_out.mean() - fake_out.mean()
        loss_ebm = self.feat_adv_loss * (cdiv_loss + reg_loss)
        return loss_ebm

    def forward(self, xyz, is_partial=True):
        z = self.encoder(xyz)
        # tanh test
        if is_partial:
            z = z + self.feat_trans.sample_langevin(z)
        pcd = self.decoder(z)
        # pcd = xyz
        return pcd, z

    def gen_params(self):
        params = list(self.encoder.parameters())
        params += list(self.decoder.parameters())
        return params

    def dis_params(self):
        params = list(self.point_disc.parameters())
        return params

    def ebm_params(self):
        params = list(self.feat_trans.parameters())
        return params


class EBM(nn.Module):
    def __init__(self, dim=1024, step_size=1, n_step=32, noise_scale=None):
        super().__init__()
        self.step_size = step_size
        self.n_step = n_step
        self.noise_scale = noise_scale
        if noise_scale is None:
            self.noise_scale = np.sqrt(step_size * 2)

        self.model = nn.Sequential(
            MLP_Res(dim, dim, dim, activation='gelu'),
            nn.Conv1d(dim, dim, 1),
            nn.GELU(),
            nn.Conv1d(dim, 1, 1)
        )

    def sample_langevin(self, z0):
        z = z0.clone().detach()
        z.requires_grad = True
        with torch.enable_grad():
            for _ in range(self.n_step):
                noise = torch.randn_like(z) * self.noise_scale
                out = self.model(z)
                grad = torch.autograd.grad(out.sum(), z, only_inputs=True)[0]
                dynamics = self.step_size * grad + noise
                z = z + dynamics
        z = z - z0
        return z.detach()

    def energy(self, z):
        return self.model(z).squeeze(-1)

    def forward(self, z):
        return self.model(z).squeeze(-1)
