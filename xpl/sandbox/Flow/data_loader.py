import matplotlib.pyplot as plt
from glob import glob
import os
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch
import random
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True





class YFCCFlowDataset(torch.utils.data.Dataset):
    def __init__(self,
                 home_dir='/data/yfcc100m/data/videos/keyframes/',
                 image_size=64,
                 random_transform=False):
        super(YFCCFlowDataset, self).__init__()
        self.home_dir = home_dir
        self.pairs = self.parse_dataset(self.home_dir)
        self.random_transform = random_transform
        self.image_size = image_size

    def parse_dataset(self, home_dir):
        all_files = sorted([y for x in os.walk(home_dir)
                            for y in glob(os.path.join(x[0], '*.jpg'))])
        dir_names = list(set([os.path.dirname(f) for f in all_files]))
        clusters = {d: [] for d in dir_names}
        [clusters[os.path.dirname(f)].append(f) for f in all_files]
        # clusters = {
        #    d: [f for f in all_files if f.startswith(d)] for d in dir_names}
        list_of_pairs = [list(zip(clusters[d][1:], clusters[d][: -1]))
                         for d in clusters.keys()]
        pairs = [item for sublist in list_of_pairs for item in sublist]

        def is_valid_pair(a, b):
            diff = int(a.split('-')[-1].split('.')[0]) - \
                int(b.split('-')[-1].split('.')[0])
            return diff <= 1 and diff >= -1
        pairs = [(a, b) for a, b in pairs if is_valid_pair(a, b)]
        symmetric_pairs = pairs + [(b, a) for (a, b) in pairs]

        return symmetric_pairs

    def __getitem__(self, index):
        first_frame_path, second_frame_path = self.pairs[index]
        first_frame = transforms.functional_pil.Image.open(first_frame_path)
        second_frame = transforms.functional_pil.Image.open(second_frame_path)

        if self.random_transform:
            # Resize
            resize = transforms.Resize(size=(2*self.image_size,
                                             2*self.image_size))
            first_frame = resize(first_frame)
            second_frame = resize(second_frame)

            # Rotate
            degree = transforms.RandomRotation.get_params(degrees=[-10, 10])
            first_frame = TF.rotate(first_frame, degree)
            second_frame = TF.rotate(second_frame, degree)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                first_frame, output_size=(self.image_size, self.image_size))
            first_frame = TF.crop(first_frame, i, j, h, w)
            second_frame = TF.crop(second_frame, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                first_frame = TF.hflip(first_frame)
                second_frame = TF.hflip(second_frame)

            # Random vertical flipping
            if random.random() > 0.5:
                first_frame = TF.vflip(first_frame)
                second_frame = TF.vflip(second_frame)

        else:
            resize = transforms.Resize(size=(self.image_size,
                                             self.image_size))
            first_frame = resize(first_frame)
            second_frame = resize(second_frame)

        first_frame = TF.to_tensor(first_frame)
        second_frame = TF.to_tensor(second_frame)

        if first_frame.shape[0] == 1:
            first_frame = torch.cat(
                [first_frame, first_frame, first_frame], dim=0)

        if second_frame.shape[0] == 1:
            second_frame = torch.cat(
                [second_frame, second_frame, second_frame], dim=0)

        return {'indices': index,
                'first_frames': first_frame,
                'second_frames': second_frame,
                }

    def __len__(self):
        return len(self.pairs)


class SintelDataset(torch.utils.data.Dataset):

    def __init__(self,
                 home_dir='/data/sintel/',
                 image_size=64,
                 random_transform=True):
        super(SintelDataset, self).__init__()
        self.home_dir = home_dir
        self.random_transform = random_transform
        self.image_size = image_size
        self.pairs = self.parse_dataset(self.home_dir)


    def parse_dataset(self, home_dir):
        all_files = sorted([y for x in os.walk(os.path.join(home_dir, 'training', 'clean'))
                            for y in glob(os.path.join(x[0], '*.png'))])
        dir_names = list(set([os.path.dirname(f) for f in all_files]))
        clusters = {d: [] for d in dir_names}
        [clusters[os.path.dirname(f)].append(f) for f in all_files]
        list_of_pairs = [list(zip(clusters[d][: -1], clusters[d][1:]))
                         for d in clusters.keys()]
        pairs = [item for sublist in list_of_pairs for item in sublist]
        split_idx = len(pairs) // 10
        if self.random_transform:
            pairs = pairs[split_idx:]
        else:
            pairs = pairs[:split_idx]
        return pairs

    def __getitem__(self, index):
        second_frame_path, first_frame_path = self.pairs[index]
        first_frame = transforms.functional_pil.Image.open(first_frame_path)
        second_frame = transforms.functional_pil.Image.open(second_frame_path)
        occlusion = transforms.functional_pil.Image.open(
            second_frame_path.replace('clean', 'occlusions'))
        flow = self.read(second_frame_path.replace(
            'clean', 'flow').replace('png', 'flo'))

        first_frame = TF.to_tensor(first_frame)
        second_frame = TF.to_tensor(second_frame)
        occlusion = TF.to_tensor(occlusion)
        flow = TF.to_tensor(flow)

        if self.random_transform:
            # Resize
            W = 2*self.image_size
            H = 2*self.image_size
            resize = transforms.Resize(size=(W, H))
            first_frame = resize(first_frame)
            second_frame = resize(second_frame)
            occlusion = resize(occlusion)
            flow[1, :, :] *= W / flow.shape[1]
            flow[0, :, :] *= H / flow.shape[2]
            flow = resize(flow)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                first_frame, output_size=(self.image_size, self.image_size))
            first_frame = TF.crop(first_frame, i, j, h, w)
            second_frame = TF.crop(second_frame, i, j, h, w)
            occlusion = TF.crop(occlusion, i, j, h, w)
            flow = TF.crop(flow, i, j, h, w)

            # Random horizontal flipping
            if False:  # random.random() > 0.5:
                first_frame = TF.hflip(first_frame)
                second_frame = TF.hflip(second_frame)
                occlusion = TF.hflip(occlusion)
                flow = TF.hflip(flow)
                flow[0, :, :] = -flow[0, :, :]

            # Random vertical flipping
            if True:  # random.random() > 0.5:
                first_frame = TF.vflip(first_frame)
                second_frame = TF.vflip(second_frame)
                occlusion = TF.vflip(occlusion)
                flow = TF.vflip(flow)
                flow[1, :, :] = -flow[1, :, :]
        else:
            W = self.image_size
            H = self.image_size
            resize = transforms.Resize(size=(W, H))
            first_frame = resize(first_frame)
            second_frame = resize(second_frame)
            occlusion = resize(occlusion)
            flow[1, :, :] *= W / flow.shape[1]
            flow[0, :, :] *= H / flow.shape[2]
            flow = resize(flow)

        flow[1, :, :] /= flow.shape[1] / 2
        flow[0, :, :] /= flow.shape[2] / 2

        if first_frame.shape[0] == 1:
            first_frame = torch.cat(
                [first_frame, first_frame, first_frame], dim=0)

        if second_frame.shape[0] == 1:
            second_frame = torch.cat(
                [second_frame, second_frame, second_frame], dim=0)

        return {'indices': index,
                'first_frames': first_frame,
                'second_frames': second_frame,
                'flows': flow
                }

    def __len__(self):
        return(len(self.pairs))

    def read(self, filename: str) -> np.ndarray:
        TAG_FLOAT = 202021.25
        assert type(filename) is str, "filename is not str %r" % str(filename)
        assert os.path.isfile(
            filename) is True, "filename does not exist %r" % str(filename)
        assert filename[-4:] == '.flo', "filename ending is not .flo %r" % filename[-4:]
        f = open(filename, 'rb')
        flo_number = np.fromfile(f, np.float32, count=1)[0]
        assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        # if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
        data = np.fromfile(f, np.float32, count=int(2*w*h))
        # Reshape data into 3D array (columns, rows, bands)
        flow = np.resize(data, (int(h), int(w), 2))
        f.close()

        return flow

    def project(self, frame, flow):
        horizontal_grid = torch.linspace(start=-1.0 + (1.0 / flow.shape[3]),
                                         end=1.0 - (1.0 / flow.shape[3]),
                                         steps=flow.shape[3]
                                         ).view(1, 1, 1, -1).expand(-1, -1, flow.shape[2], -1)
        vertical_grid = torch.linspace(start=-1.0 + (1.0 / flow.shape[2]),
                                       end=1.0 - (1.0 / flow.shape[2]),
                                       steps=flow.shape[2]
                                       ).view(1, 1, -1, 1).expand(-1, -1, -1, flow.shape[3])

        grid = torch.cat(
            tensors=[horizontal_grid, vertical_grid], dim=1).to(flow.device)
        grid = (grid + flow).permute(0, 2, 3, 1)

        return torch.nn.functional.grid_sample(input=frame,
                                               grid=grid,
                                               mode='bilinear',
                                               padding_mode='border',
                                               align_corners=False)


if __name__ == "__main__":
    s = SintelDataset(random_transform=False)

    batch = next(iter(s))
    first = batch['first_frames']
    second = batch['second_frames']
    o = batch['occlusion']
    flow = batch['flow']

    proj = s.project(second.unsqueeze(0), flow.unsqueeze(0)).squeeze(0)

    u = (first - proj).abs()
    v0 = u[0, :, :] * (1 - o.squeeze(0))
    v1 = u[1, :, :] * (1 - o.squeeze(0))
    v2 = u[2, :, :] * (1 - o.squeeze(0))

    v = torch.cat([v0.unsqueeze(0),
                   v1.unsqueeze(0),
                   v2.unsqueeze(0)], dim=0)

    plt.subplot(3, 2, 1)
    plt.imshow(first.permute(1, 2, 0).numpy())
    plt.subplot(3, 2, 2)
    plt.imshow(second.permute(1, 2, 0).numpy())
    plt.subplot(3, 2, 3)
    plt.imshow(proj.permute(1, 2, 0).numpy())
    plt.subplot(3, 2, 4)
    plt.imshow(u.permute(1, 2, 0).numpy())
    plt.subplot(3, 2, 5)
    plt.imshow(o.permute(1, 2, 0).numpy())
    plt.subplot(3, 2, 6)
    plt.imshow(v.permute(1, 2, 0).numpy())
    print(v.max())
    plt.show()
