
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.utils.data
import torch.nn.functional as F
from dataset import NerfDataset


def to_cartesian(theta_phi):
    return torch.stack([torch.sin(theta_phi[:, 0]) * torch.cos(theta_phi[:, 1]),
                        torch.sin(theta_phi[:, 0]) * torch.sin(theta_phi[:, 1]),
                        torch.cos(theta_phi[:, 0])], axis=1)

def to_rodriguet(r, t):
    phi_skew = torch.stack([torch.cat([torch.zeros(1, device=r.device), -r[2:3], r[1:2]]),
                            torch.cat([r[2:3], torch.zeros(1, device=r.device), -r[0:1]]),
                            torch.cat([-r[1:2], r[0:1], torch.zeros(1, device=r.device)])], dim=0)
    alpha = r.norm() + 1e-15
    R = torch.eye(3, device=r.device) + (torch.sin(alpha) / alpha) * phi_skew + (
            (1 - torch.cos(alpha)) / alpha ** 2) * (phi_skew @ phi_skew)
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)
    c2w = torch.cat([c2w, torch.tensor([[0., 0., 0., 1.]], device=c2w.device)], dim=0)
    return c2w


class NerfModel(nn.Module):
    def __init__(self, embedding_dim_pos=10, hidden_dim=256):
        super(NerfModel, self).__init__()

        self.block1 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )

        self.block2 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + 3 + hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim + 1), )

        self.block3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), )
        self.block4 = nn.Sequential(nn.Linear(hidden_dim // 2, 75), )

        self.embedding_dim_pos = embedding_dim_pos
        self.relu = nn.ReLU()

        self.bandwidth = nn.Parameter(torch.zeros((1, 25)))
        self.p = nn.Parameter(torch.randn((25, 2)))

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos)
        h = self.block1(emb_x)
        tmp = self.block2(torch.cat((h, emb_x), dim=1))
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1])
        h = self.block3(h)
        k = self.block4(h).reshape(o.shape[0], 25, 3)

        c = (k * torch.exp((self.bandwidth.unsqueeze(-1) * to_cartesian(self.p).unsqueeze(0) * d.unsqueeze(1)))).sum(1)

        return torch.sigmoid(c), sigma

# SH_C0 = 0.28209479177387814
# SH_C1 = 0.4886025119029199
# SH_C2 = [
#     1.0925484305920792,
#     -1.0925484305920792,
#     0.31539156525252005,
#     -1.0925484305920792,
#     0.5462742152960396
# ]
# SH_C3 = [
#     -0.5900435899266435,
#     2.890611442640554,
#     -0.4570457994644658,
#     0.3731763325901154,
#     -0.4570457994644658,
#     1.445305721320277,
#     -0.5900435899266435
# ]
# SH_C4 = [
#     2.5033429417967046,
#     -1.7701307697799304,
#     0.9461746957575601,
#     -0.6690465435572892,
#     0.10578554691520431,
#     -0.6690465435572892,
#     0.47308734787878004,
#     -1.7701307697799304,
#     0.6258357354491761,
# ]

# def eval_spherical_function(basis_dim, k, d):  
#     x, y, z = d[..., 0:1], d[..., 1:2], d[..., 2:3]  
#     result = SH_C0 * k[..., 0]
#     if basis_dim > 1:
#         result += -SH_C1 * y * k[..., 1]
#         result += SH_C1 * z * k[..., 2]
#         result += -SH_C1 * x * k[..., 3]
#         if basis_dim > 4:
#             xx, yy, zz = x * x, y * y, z * z
#             xy, yz, xz = x * y, y * z, x * z
#             result += SH_C2[0] * xy * k[..., 4]
#             result += SH_C2[1] * yz * k[..., 5]
#             result += SH_C2[2] * (2.0 * zz - xx - yy) * k[..., 6]
#             result += SH_C2[3] * xz * k[..., 7]
#             result += SH_C2[4] * (xx - yy) * k[..., 8]

#             if basis_dim > 9:
#                 result += SH_C3[0] * y * (3 * xx - yy) * k[..., 9]
#                 result += SH_C3[1] * xy * z * k[..., 10]
#                 result += SH_C3[2] * y * (4 * zz - xx - yy) * k[..., 11]
#                 result += SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * k[..., 12]
#                 result += SH_C3[4] * x * (4 * zz - xx - yy) * k[..., 13]
#                 result += SH_C3[5] * z * (xx - yy) * k[..., 14]
#                 result += SH_C3[6] * x * (xx - 3 * yy) * k[..., 15]

#                 if basis_dim > 16:
#                     result += SH_C4[0] * xy * (xx - yy) * k[..., 16]
#                     result += SH_C4[1] * yz * (3 * xx - yy) * k[..., 17]
#                     result += SH_C4[2] * xy * (7 * zz - 1) * k[..., 18]
#                     result += SH_C4[3] * yz * (7 * zz - 3) * k[..., 19]
#                     result += SH_C4[4] * (zz * (35 * zz - 30) + 3) * k[..., 20]
#                     result += SH_C4[5] * xz * (7 * zz - 3) * k[..., 21]
#                     result += SH_C4[6] * (xx - yy) * (7 * zz - 1) * k[..., 22]
#                     result += SH_C4[7] * xz * (xx - 3 * yy) * k[..., 23]
#                     result += SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * k[..., 24]
#     return result

# def gradient(x, y):
#     x.requires_grad_(True)
#     d_output = torch.ones_like(y, requires_grad=False, device=y.device)
#     gradients = torch.autograd.grad(
#         outputs=y,
#         inputs=x,
#         grad_outputs=d_output,
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True)[0]
#     return gradients.unsqueeze(1)

# class NerfModel(nn.Module):
#     def __init__(self, N=256):
#         """
#         :param N
#         """
#         super(NerfModel, self).__init__()
#         self.voxel_grid = nn.Parameter(torch.ones((N + 1, N + 1, N + 1, 27 + 1)) / 100)
#         self.N = N

#     def get_coord(self, x):
#         ratio = (2 * self.scale / self.N)
#         return x / ratio + self.N / 2

#     def total_variation_loss(self, weight_sh, weight_sigma):
#         h_grid, w_grid, d_grid, c_grid = self.voxel_grid.size()

#         tv_h = torch.pow(self.voxel_grid[1:,:,:,1:]-self.voxel_grid[:-1,:,:,1:], 2).sum()
#         tv_w = torch.pow(self.voxel_grid[:,1:,:,1:]-self.voxel_grid[:,:-1,:,1:], 2).sum()
#         tv_d = torch.pow(self.voxel_grid[:,:,1:,1:]-self.voxel_grid[:,:,:-1,1:], 2).sum()
#         loss_sh = weight_sh*(tv_h+tv_w+tv_d)/((c_grid - 1)*h_grid*w_grid*d_grid)

#         tv_h_sigma = torch.pow(self.voxel_grid[1:,:,:,0]-self.voxel_grid[:-1,:,:,0], 2).sum()
#         tv_w_sigma = torch.pow(self.voxel_grid[:,1:,:,0]-self.voxel_grid[:,:-1,:,0], 2).sum()
#         tv_d_sigma = torch.pow(self.voxel_grid[:,:,1:,0]-self.voxel_grid[:,:,:-1,0], 2).sum()
#         loss_sigma = weight_sigma*(tv_h_sigma+tv_w_sigma+tv_d_sigma)/(1*h_grid*w_grid*d_grid)

#         return loss_sh + loss_sigma

#     def forward(self, x, d):
#         tmp = torch.nn.functional.grid_sample(self.voxel_grid.unsqueeze(0).permute(0, 4, 1, 2, 3), x.unsqueeze(0).unsqueeze(0).unsqueeze(0),\
#                 mode='bilinear', padding_mode="zeros", align_corners=False).permute(0, 2, 3, 4, 1).squeeze(0).squeeze(0).squeeze(0)

#         return eval_spherical_function(9, tmp[:, 1:].reshape(-1, 3, 9), d), torch.nn.functional.relu(tmp[:, 0])


def sample_batch(camera_extrinsic, camera_intrinsic, image, batch_size=None, sample_all=False):
    if sample_all:
        u, v = np.meshgrid(np.linspace(0, image.shape[1] - 1, image.shape[1], dtype=int), np.linspace(0, image.shape[0] - 1, image.shape[0], dtype=int))
        u = torch.from_numpy(u.reshape(-1)).to(camera_intrinsic.device)
        v = torch.from_numpy(v.reshape(-1)).to(camera_intrinsic.device)
    else:
        u = torch.randint(image.shape[1], (batch_size,), device=camera_intrinsic.device)
        v = torch.randint(image.shape[0], (batch_size,), device=camera_intrinsic.device)

    rays_d_cam = torch.cat([((u - camera_intrinsic[0, 2]) / camera_intrinsic[0,0]).unsqueeze(-1),
                            (-(v - camera_intrinsic[1,2]) / camera_intrinsic[1,1]).unsqueeze(-1),
                            torch.ones_like(u).unsqueeze(-1)], dim=-1)
    
    rays_d_world = torch.matmul(camera_extrinsic[:3, :3].T.view(1, 3, 3), rays_d_cam.unsqueeze(2)).squeeze(2)
    rays_o_world = torch.matmul(-camera_extrinsic[:3, :3].T.view(1, 3, 3), camera_extrinsic[:3, 3].view(1, 3).expand_as(rays_d_world).unsqueeze(2)).squeeze(2)

    d_norm = F.normalize(rays_d_world, p=2, dim=1)
    rays_o_new = rays_o_world + 0.01*d_norm

    ground_truth_px_values = None

    if batch_size and batch_size > 0:
        ground_truth_px_values = image[(v, u)]

    return rays_o_new, d_norm, ground_truth_px_values


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)


def render_rays(nerf_model, ray_origins, ray_directions, nb_bins=192):
    
    colors, sigma = nerf_model(w.reshape(-1, 3), nrd.reshape(-1, 3))
    alpha = 1 - torch.exp(-sigma.reshape(w.shape[:-1]) * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    return (weights * colors.reshape(w.shape)).sum(dim=1) + 1 - weights.sum(-1).sum(-1).unsqueeze(-1)


def train(nerf_model, optimizer, scheduler, data_loader, device='cuda', nb_epochs=int(1e5), nb_bins=192):
    # training_loss = []
    for _ in tqdm(range(nb_epochs)):
        for batch in data_loader:
            training_image = batch["img"].squeeze(0).to(device)
            camera_intrinsic = batch["K"].squeeze(0).to(device)
            camera_extrinsic = batch["Rt"].squeeze(0).to(device)

            o_norm, d_norm, ground_truth_px_values = \
                sample_batch(camera_extrinsic, camera_intrinsic, training_image, batch_size=2048)

            regenerated_px_values = render_rays(nerf_model, o_norm, d_norm, nb_bins=nb_bins)
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()

            # print(loss)

            # print(ground_truth_px_values)
            # print(" ===> ",regenerated_px_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # training_loss.append(loss.item())
        scheduler.step()

    # return training_loss


@torch.no_grad()
def test(model, camera_intrinsic, camera_extrinsic, chunk_size=1024, nb_bins=192, H=400, W=400):
    o_norm, d_norm, _ = \
                sample_batch(camera_extrinsic, camera_intrinsic, torch.ones((H, W, 3)), None, sample_all=True)
    data = []
    part = int((W*H)/chunk_size)
    for i in range(part):
        if i < part-1:
            o_norm_ = o_norm[i * chunk_size: (i + 1) * chunk_size].to(camera_intrinsic.device)
            d_norm_ = d_norm[i * chunk_size: (i + 1) * chunk_size].to(camera_intrinsic.device)
        else:
            o_norm_ = o_norm[i * chunk_size:].to(camera_intrinsic.device)
            d_norm_ = d_norm[i * chunk_size:].to(camera_intrinsic.device)
        regenerated_px_values = render_rays(model, o_norm_, d_norm_, nb_bins=nb_bins)
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)
    plt.imshow(img)
    plt.savefig(f'Imgs/novel_view.png', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    device = 'cuda'

    # dataset = NerfDataset("../blended_mvs/5a3ca9cb270f0e3f14d0eddb/", 'train')
    dataset = NerfDataset("./lego/", 'train')
    data_loader = DataLoader(dataset, 1, shuffle=False, num_workers=8, drop_last=True)
    
    model = NerfModel().to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)

    train(model, model_optimizer, scheduler, data_loader, device=device, nb_epochs=50, nb_bins=192)
    
    for idx, batch in tqdm(enumerate(data_loader)):
        if idx == 10:
            camera_intrinsic = batch["K"].squeeze(0).to(device)
            camera_extrinsic = batch["Rt"].squeeze(0).to(device)
            test(model, camera_intrinsic, camera_extrinsic, H=800, W=800)