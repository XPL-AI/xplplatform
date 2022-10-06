import numpy as np
import torch
import torch.nn.functional as F


class L1_Laplacian_Pyramid_Loss(torch.nn.Module):

    def __init__(self, num_channels):
        super(L1_Laplacian_Pyramid_Loss, self).__init__()

        self.num_channels = num_channels
        self.pyramid_levels = 3
        self.kernel_size = 5
        self.sigma = 1.0
        self.kernel = self.init_kernel()

    def init_kernel(self):
        assert(self.kernel_size % 2 == 1),\
            'Kernel size should be odd but it is %d' % (self.kernel_size)

        grid = np.float32(np.mgrid[0:self.kernel_size,
                                   0:self.kernel_size].T)

        def gaussian(x):
            return np.exp((x - self.kernel_size // 2)**2 / (-2 * self.sigma**2))**2

        kernel = np.sum(gaussian(grid), axis=2)
        kernel /= np.sum(kernel)
        kernel = np.tile(kernel, (self.num_channels, 1, 1))
        kernel = torch.FloatTensor(kernel)
        kernel = kernel.unsqueeze(1)
        return kernel

    def laplacian_pyramid(self, image):
        current = image
        pyramid = []

        for _ in range(self.pyramid_levels):
            filtered = self.conv_gauss(current)
            diff = current - filtered
            pyramid.append(diff)
            current = F.avg_pool2d(filtered, 2)

        pyramid.append(current)
        return pyramid

    def downscale_pyramid(self, image):
        current = image
        pyramid = []
        for _ in range(self.pyramid_levels):
            pyramid.append(current)
            current = F.avg_pool2d(current, 2)
        pyramid.append(current)
        return pyramid

    def conv_gauss(self, image):
        pad = tuple([self.kernel_size // 2] * 4)
        image = F.pad(image, pad, mode='replicate')

        return F.conv2d(input=image,
                        weight=self.kernel.to(image.device),
                        groups=self.num_channels)

    def forward(self,
                predictions,
                targets,
                occlusions):
        pyramid_prediction = self.laplacian_pyramid(predictions)
        pyramid_target = self.laplacian_pyramid(targets.detach())
        if occlusions is not None:
            pyramid_occlusion = self.downscale_pyramid(occlusions.detach())
            losses = [((p - t).abs() * (1.0 - torch.exp(o).expand_as(p))).mean(-1).mean(-1).mean(-1)
                      for p, t, o in zip(pyramid_prediction, pyramid_target, pyramid_occlusion)]
        else:
            losses = [((p - t).abs()).mean(-1).mean(-1).mean(-1)
                      for p, t in zip(pyramid_prediction, pyramid_target)]

        return sum(losses)


class UnsupervisedOpticalFlowLoss(torch.nn.Module):

    def __init__(self):
        super(UnsupervisedOpticalFlowLoss, self).__init__()
        self.l1_loss = L1_Laplacian_Pyramid_Loss(num_channels=3)

    def forward(self,
                first_frames,
                second_frames,
                flows_1_to_2,
                flows_2_to_1,
                occlusions_1_to_2,
                occlusions_2_to_1):

        projected_second_frames = self.project(frames=first_frames,
                                               flows=flows_1_to_2)
        projected_first_frames = self.project(frames=second_frames,
                                              flows=flows_2_to_1)

        consistency_loss = sum((
            self.l1_loss(predictions=projected_second_frames,
                         targets=second_frames,
                         occlusions=occlusions_1_to_2 if torch.exp(occlusions_1_to_2).mean() < 0.1 else None),
            self.l1_loss(predictions=projected_first_frames,
                         targets=first_frames,
                         occlusions=occlusions_2_to_1 if torch.exp(occlusions_2_to_1).mean() < 0.1 else None)
        ))

        smoothness_loss = sum((
            self.smoothness_loss(flows_1_to_2, first_frames),
            self.smoothness_loss(flows_2_to_1, second_frames),
        ))

        sparsity_loss = sum((
            self.sparsity_loss(occlusion_prob=torch.exp(occlusions_1_to_2),
                               ro_hat=0.1),
            self.sparsity_loss(occlusion_prob=torch.exp(occlusions_2_to_1),
                               ro_hat=0.1)

        ))

        return {
            'loss': {
                'consistency': consistency_loss,
                'smoothness': smoothness_loss,
                'sparsity': sparsity_loss
            }
        }

    def sparsity_loss(self,
                      occlusion_prob,
                      ro_hat: float = 0.2):

        ro = occlusion_prob.mean(-1).mean(-1).mean(-1)
        loss = -(ro_hat * torch.log(ro / ro_hat) +
                 (1.0-ro_hat) * torch.log((1.0 - ro) / (1.0 - ro_hat)))
        loss[ro < ro_hat] = 0.0
        return loss / 10.0

    def get_surface_gradients(self, surface):
        gradient_accross_x = surface[:, :, :, 1:] - surface[:, :, :, :-1]
        gradient_accross_y = surface[:, :, 1:, :] - surface[:, :, :-1, :]
        gradient_norm_accross_x = torch.sqrt(1e-5 +
                                             (gradient_accross_x ** 2).sum(dim=1, keepdim=True))
        gradient_norm_accross_y = torch.sqrt(1e-5 +
                                             (gradient_accross_y ** 2).sum(dim=1, keepdim=True))

        return gradient_norm_accross_x, gradient_norm_accross_y

    def smoothness_loss(self, flows, frames):
        flows_norm = torch.sqrt(flows[:, 0:1, :, :]**2 + flows[:, 1:2, :, :]**2)
        flows_norm_x, flows_norm_y = self.get_surface_gradients(flows_norm)
        frames_norm_x, frames_norm_y = self.get_surface_gradients(frames)

        return sum((
            (torch.abs(flows_norm_x) * torch.exp(- torch.abs(frames_norm_x))).mean(-1).mean(-1).mean(-1),
            (torch.abs(flows_norm_y) * torch.exp(- torch.abs(frames_norm_y))).mean(-1).mean(-1).mean(-1)
        ))

    def project(self, frames, flows):
        horizontal_grid = torch.linspace(start=-1.0 + (1.0 / flows.shape[3]),
                                         end=1.0 - (1.0 / flows.shape[3]),
                                         steps=flows.shape[3]
                                         ).view(1, 1, 1, -1).expand(-1, -1, flows.shape[2], -1)
        vertical_grid = torch.linspace(start=-1.0 + (1.0 / flows.shape[2]),
                                       end=1.0 - (1.0 / flows.shape[2]),
                                       steps=flows.shape[2]
                                       ).view(1, 1, -1, 1).expand(-1, -1, -1, flows.shape[3])

        grid = torch.cat(
            tensors=[horizontal_grid, vertical_grid], dim=1).to(flows.device)
        grid = (grid + flows).permute(0, 2, 3, 1)

        return torch.nn.functional.grid_sample(input=frames,
                                               grid=grid,
                                               mode='bilinear',
                                               padding_mode='border',
                                               align_corners=False)


class SupervisedOpticalFlowLoss(torch.nn.Module):

    def __init__(self):
        super(SupervisedOpticalFlowLoss, self).__init__()
        self.l1_loss = L1_Laplacian_Pyramid_Loss(num_channels=2)

    def forward(self,
                first_frames,
                second_frames,
                predicted_flows,
                target_flows,
                predicted_occlusions):

        flow_loss = self.l1_loss(predictions=predicted_flows,
                                 targets=target_flows,
                                 occlusions=None) * 10

        projected_second_frames = self.project(frames=first_frames,
                                               flows=target_flows)
        residuals = (projected_second_frames - second_frames).abs()
        target_occlusions = (residuals.sum(dim=1, keepdim=True) > 0.1).float().detach().clamp(min=0.0, max=1.0 - 1e-8)

        occlusion_loss = F.binary_cross_entropy_with_logits(input=predicted_occlusions,
                                                           target=target_occlusions,
                                                           reduction='none').mean(-1).mean(-1).mean(-1)

        return {
            'loss': {
                'flow': flow_loss,
                'occlusion': occlusion_loss
            }
        }

    def project(self, frames, flows):
        horizontal_grid = torch.linspace(start=-1.0 + (1.0 / flows.shape[3]),
                                         end=1.0 - (1.0 / flows.shape[3]),
                                         steps=flows.shape[3]
                                         ).view(1, 1, 1, -1).expand(-1, -1, flows.shape[2], -1)
        vertical_grid = torch.linspace(start=-1.0 + (1.0 / flows.shape[2]),
                                       end=1.0 - (1.0 / flows.shape[2]),
                                       steps=flows.shape[2]
                                       ).view(1, 1, -1, 1).expand(-1, -1, -1, flows.shape[3])

        grid = torch.cat(
            tensors=[horizontal_grid, vertical_grid], dim=1).to(flows.device)
        grid = (grid + flows).permute(0, 2, 3, 1)

        return torch.nn.functional.grid_sample(input=frames,
                                               grid=grid,
                                               mode='bilinear',
                                               padding_mode='border',
                                               align_corners=False)
