import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import torch.autograd.profiler as profiler


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

def eval_spherical_function(basis_dim, k, d):  
    x, y, z = d[..., 0:1], d[..., 1:2], d[..., 2:3]  
    result = SH_C0 * k[..., 0]
    if basis_dim > 1:
        result += -SH_C1 * y * k[..., 1]
        result += SH_C1 * z * k[..., 2]
        result += -SH_C1 * x * k[..., 3]
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result += SH_C2[0] * xy * k[..., 4]
            result += SH_C2[1] * yz * k[..., 5]
            result += SH_C2[2] * (2.0 * zz - xx - yy) * k[..., 6]
            result += SH_C2[3] * xz * k[..., 7]
            result += SH_C2[4] * (xx - yy) * k[..., 8]

            if basis_dim > 9:
                result += SH_C3[0] * y * (3 * xx - yy) * k[..., 9]
                result += SH_C3[1] * xy * z * k[..., 10]
                result += SH_C3[2] * y * (4 * zz - xx - yy) * k[..., 11]
                result += SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * k[..., 12]
                result += SH_C3[4] * x * (4 * zz - xx - yy) * k[..., 13]
                result += SH_C3[5] * z * (xx - yy) * k[..., 14]
                result += SH_C3[6] * x * (xx - 3 * yy) * k[..., 15]

                if basis_dim > 16:
                    result += SH_C4[0] * xy * (xx - yy) * k[..., 16]
                    result += SH_C4[1] * yz * (3 * xx - yy) * k[..., 17]
                    result += SH_C4[2] * xy * (7 * zz - 1) * k[..., 18]
                    result += SH_C4[3] * yz * (7 * zz - 3) * k[..., 19]
                    result += SH_C4[4] * (zz * (35 * zz - 30) + 3) * k[..., 20]
                    result += SH_C4[5] * xz * (7 * zz - 3) * k[..., 21]
                    result += SH_C4[6] * (xx - yy) * (7 * zz - 1) * k[..., 22]
                    result += SH_C4[7] * xz * (xx - 3 * yy) * k[..., 23]
                    result += SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * k[..., 24]
    return result

def gradient(x, y):
    x.requires_grad_(True)
    d_output = torch.ones_like(y, requires_grad=False, device=y.device)
    gradients = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=d_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return gradients.unsqueeze(1)

class NerfModel(nn.Module):
    def __init__(self, N=1024, scale=1.5):
        """
        :param N
        :param scale: The maximum absolute value among all coordinates for objects in the scene
        """
        super(NerfModel, self).__init__()

        self.edges = nn.Parameter(torch.ones((N, 3, 8)) / 100)
        self.corners = nn.Parameter(torch.ones(10, 8) / 100)
        self.scale = scale
        self.N = N

    def get_coord(self, x):
        ratio = (2 * self.scale / self.N)
        return x / ratio + self.N / 2

    def total_variation_loss(self, weight_sh, weight_sigma):
        h_grid, w_grid, d_grid, c_grid = self.voxel_grid.size()

        tv_h = torch.pow(self.voxel_grid[1:,:,:,1:]-self.voxel_grid[:-1,:,:,1:], 2).sum()
        tv_w = torch.pow(self.voxel_grid[:,1:,:,1:]-self.voxel_grid[:,:-1,:,1:], 2).sum()
        tv_d = torch.pow(self.voxel_grid[:,:,1:,1:]-self.voxel_grid[:,:,:-1,1:], 2).sum()
        loss_sh = weight_sh*(tv_h+tv_w+tv_d)/((c_grid - 1)*h_grid*w_grid*d_grid)

        tv_h_sigma = torch.pow(self.voxel_grid[1:,:,:,0]-self.voxel_grid[:-1,:,:,0], 2).sum()
        tv_w_sigma = torch.pow(self.voxel_grid[:,1:,:,0]-self.voxel_grid[:,:-1,:,0], 2).sum()
        tv_d_sigma = torch.pow(self.voxel_grid[:,:,1:,0]-self.voxel_grid[:,:,:-1,0], 2).sum()
        loss_sigma = weight_sigma*(tv_h_sigma+tv_w_sigma+tv_d_sigma)/(1*h_grid*w_grid*d_grid)

        return loss_sh + loss_sigma

    def forward(self, x, d):
        # with profiler.record_function("FORWARD PASS"):
        color = torch.zeros_like(x)
        sigma = torch.zeros((x.shape[0]), device=x.device)
        mask = (x[:, 0] < self.scale) & (x[:, 1] > -self.scale) & (x[:, 2] > -self.scale) \
            & (x[:, 0] > -self.scale) & (x[:, 1] < self.scale) & (x[:, 2] < self.scale)
        indices = torch.linspace(0., mask.shape[0] - 1., mask.shape[0], device=x.device).long() * mask
        
        tmp = torch.nn.functional.grid_sample(self.voxel_grid.unsqueeze(0).permute(0, 4, 1, 2, 3), x[indices].unsqueeze(0).unsqueeze(0).unsqueeze(0) / self.scale,\
                mode='bilinear', padding_mode="zeros", align_corners=False).permute(0, 2, 3, 4, 1).squeeze(0).squeeze(0).squeeze(0)

        sigma[indices] = torch.nn.functional.relu(tmp[:, 0])
        # sigmoid = torch.sigmoid(tmp[:, 0])
        # alpha = torch.nn.functional.relu((sigmoid[:-1, :] - sigmoid[1:, :])/sigmoid[:-1, :])
        k = tmp[:, 1:]
        color[indices] = eval_spherical_function(9, k.reshape(-1, 3, 9), d[indices])

        return color, sigma


@torch.no_grad()
def test(hn, hf, dataset, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400):
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

    data = []
    for i in range(int(np.ceil(H / chunk_size))):
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    plt.figure()
    plt.imshow(img)
    plt.savefig(f'Imgs/img_{img_index}.png', bbox_inches='tight')
    plt.close()


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)


def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.ones((ray_origins.shape[0], 1), device=device)*1e10), -1)

    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)  # [batch_size, nb_bins, 3]
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)

    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)  # Pixel values
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background
    return c + 1 - weight_sum.unsqueeze(-1)


def train(nerf_model, optimizer, scheduler, data_loader, device='cpu', hn=0, hf=1, nb_epochs=int(1e5),
          nb_bins=192):
    # training_loss = []
    torch.cuda.synchronize()
    for _ in range(nb_epochs):
        for batch in tqdm(data_loader):
            ray_origins = batch[:, :3]#.to(device)
            ray_directions = batch[:, 3:6]#.to(device)
            ground_truth_px_values = batch[:, 6:]#.to(device)

            # with profiler.profile(with_stack=True, profile_memory=True) as prof:
            regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
            # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
            
            loss = torch.nn.functional.mse_loss(ground_truth_px_values, regenerated_px_values) + nerf_model.total_variation_loss(5e-3, 5e-3)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # training_loss.append(loss.item())

        scheduler.step()
    # return training_loss


if __name__ == "__main__":
    device = 'cuda'
    training_dataset = torch.from_numpy(np.load('nerf_datasets/training_data.pkl', allow_pickle=True)).to(device)
    testing_dataset = torch.from_numpy(np.load('nerf_datasets/testing_data.pkl', allow_pickle=True)).to(device)
    model = NerfModel(N=256).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)

    data_loader = DataLoader(training_dataset, batch_size=2048, shuffle=True)
    train(model, model_optimizer, scheduler, data_loader, nb_epochs=1, device=device, hn=2, hf=6, nb_bins=192)
    for img_index in [0, 60, 120, 180]:
        test(2, 6, testing_dataset, img_index=img_index, nb_bins=192, H=400, W=400)