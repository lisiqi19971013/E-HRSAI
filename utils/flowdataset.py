import glob
from PIL import Image
from torch.utils import data
import os
import torch
import numpy as np
import cv2
from torchvision import transforms
import random


class dataset_flow(data.Dataset):
    def __init__(self, train=True, arg=False):
        if train:
            self.file = '/home/lisiqi/data/DeOccDepthEstimation(DODE)/total.txt'
        else:
            self.file = '/home/lisiqi/data/DeOccDepthEstimation(DODE)/total.txt'

        self.folder = []
        self.dim = (260, 260)
        self.nb_of_bin = 5

        with open(self.file, 'r') as f:
            for line in f.readlines():
                p1 = line.strip('\n')
                self.folder.append(p1)

        self.train = (train & arg)
        self.arg = arg

    def __getitem__(self, idx):
        # data = np.load(os.path.join(self.folder[idx], 'data', 'flow.npy'))
        # data = torch.from_numpy(data)
        # I0 = transforms.ToTensor()(Image.open(os.path.join(self.folder[idx], 'data', '0.png')))
        # I1 = transforms.ToTensor()(Image.open(os.path.join(self.folder[idx], 'data', '1.png')))
        # I2 = transforms.ToTensor()(Image.open(os.path.join(self.folder[idx], 'data', '2.png')))
        # event01 = data[9:9+self.nb_of_bin, ...][:,2:-2,2:-2]
        # event21 = data[9+self.nb_of_bin:, ...][:,2:-2,2:-2]
        # return I0, I1, I2, event01, event21

        event = np.load(os.path.join(self.folder[idx], 'events.npy'))
        mid = int(glob.glob(os.path.join(self.folder[idx], 'frame_*_1.jpg'))[0].split('_')[-2])
        ts = np.load(os.path.join(self.folder[idx], 'ts.npy'))
        frames = glob.glob(os.path.join(self.folder[idx], 'frame_*.jpg'))
        frames.sort()

        I0 = transforms.ToTensor()(Image.open(frames[mid-1]))
        I1 = transforms.ToTensor()(Image.open(frames[mid]))
        I2 = transforms.ToTensor()(Image.open(frames[mid+1]))

        event01 = event[(event[:, 0] >= ts[mid-1][0]) & (event[:, 0] <= ts[mid][1]), :].astype(np.long)
        event12 = event[(event[:, 0] >= ts[mid][0]) & (event[:, 0] <= ts[mid+1][1]), :].astype(np.long)
        event21 = self.reverse(event12)

        event01 = self.event2vox(event01)
        event21 = self.event2vox(event21)

        # I0, I1, I2, event01, event21 = self.data_augmentation(I0, I1, I2, event01, event21, train=self.train)
        return I0, I1, I2, event01, event21, self.folder[idx]


    def __len__(self):
        return len(self.folder)

    def reverse(self, event):
        if len(event) == 0:
            return
        event[:, 0] = (event[-1,0] - event[:, 0])
        event[:, -1] = -event[:, -1]
        event = np.copy(np.flipud(event))
        return event

    def event2vox(self, event):
        event = torch.from_numpy(event).float()
        H, W = self.dim

        voxel_grid = torch.zeros(self.nb_of_bin, H, W, dtype=torch.float32, device='cpu')
        vox = voxel_grid.ravel()

        t, x, y, p = event.t()
        t = t.long()
        time_max = t.max()
        time_min = t.min()

        t = (t-time_min) * (self.nb_of_bin - 1) / (time_max-time_min)
        t = t.float()
        left_t, right_t = t.floor(), t.floor()+1
        left_x, right_x = x.float().floor(), x.float().floor()+1
        left_y, right_y = y.float().floor(), y.float().floor()+1

        for lim_x in [left_x, right_x]:
            for lim_y in [left_y, right_y]:
                for lim_t in [left_t, right_t]:
                    mask = (0 <= lim_x) & (0 <= lim_y) & (0 <= lim_t) & (lim_x <= W-1) & (lim_y <= H-1) & (lim_t <= self.nb_of_bin-1)
                    lin_idx = lim_x.long() + lim_y.long() * W + lim_t.long() * W * H
                    weight = p * (1-(lim_x-x).abs()) * (1-(lim_y-y).abs()) * (1-(lim_t-t).abs())
                    vox.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())

        return voxel_grid

    def data_augmentation(self, I0, I1, I2, event01, event21, crop_size=(256, 256), train=True):
        if train:
            args = transforms.Compose([transforms.RandomCrop(crop_size), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
        else:
            args = transforms.Compose([transforms.CenterCrop(crop_size)])
        x = torch.cat([I0, I1, I2, event01, event21], dim=0)
        x = args(x)
        I0 = x[:3, ...]
        I1 = x[3:6, ...]
        I2 = x[6:9, ...]
        event01 = x[9:9+self.nb_of_bin, ...]
        event21 = x[9+self.nb_of_bin:, ...]
        return I0, I1, I2, event01, event21


class dataset_efsai_flow(data.Dataset):
    def __init__(self, train=True, arg=False):
        if train:
            self.file = '/home/lisiqi/data/EF-SAI-Dataset/train.txt'
        else:
            self.file = '/home/lisiqi/data/EF-SAI-Dataset/test.txt'

        self.event_file, self.gt_file, self.img_file = [], [], []
        self.dim = (260, 346)
        self.nb_of_bin = 5

        with open(self.file, 'r') as f:
            for line in f.readlines():
                p1 = line.strip('\n')
                self.event_file.append(p1.split(' ')[0].replace('_refocus', ''))
                self.img_file.append(p1.split(' ')[1].replace('_refocus', ''))
                self.gt_file.append(p1.split(' ')[2])

        self.train = (train & arg)
        self.arg = arg

    def __getitem__(self, idx):
        # data = np.load(os.path.join(self.folder[idx], 'data', 'flow.npy'))
        # data = torch.from_numpy(data)
        # I0 = transforms.ToTensor()(Image.open(os.path.join(self.folder[idx], 'data', '0.png')))
        # I1 = transforms.ToTensor()(Image.open(os.path.join(self.folder[idx], 'data', '1.png')))
        # I2 = transforms.ToTensor()(Image.open(os.path.join(self.folder[idx], 'data', '2.png')))
        # event01 = data[9:9+self.nb_of_bin, ...][:,2:-2,2:-2]
        # event21 = data[9+self.nb_of_bin:, ...][:,2:-2,2:-2]
        # return I0, I1, I2, event01, event21

        FrameData = np.load(self.img_file[idx], allow_pickle=True).item()
        EventData = np.load(self.event_file[idx], allow_pickle=True).item()

        ts = FrameData['time_stamp']
        t_ref = FrameData['ref_t']
        t_ref -= ts
        mid = np.where(t_ref == 0)[0][0]

        t0 = ts.min()
        ts -= t0
        ts *= 1e6
        dt = 10000

        # id_start = mid-8
        # id_end = mid+9

        event = np.stack([EventData['events']['t'].astype(np.float64), EventData['events']['x'].astype(np.float64),
                          EventData['events']['y'].astype(np.float64), EventData['events']['p'].astype(np.float64)]).T
        event[:, 0] -= t0
        event[:, 0] *= 1e6

        event01 = event[(event[:, 0] >= ts[mid-1]-dt) & (event[:, 0] <= ts[mid]+dt), :].astype(np.long)
        event12 = event[(event[:, 0] >= ts[mid]-dt) & (event[:, 0] <= ts[mid+1]+dt), :].astype(np.long)

        # event0 = event[(event[:, 0] >= ts[id_start-1]-dt) & (event[:, 0] <= ts[id_end-2]+dt), :].astype(np.long)
        # event1 = event[(event[:, 0] >= ts[id_start]-dt) & (event[:, 0] <= ts[id_end-1]+dt), :].astype(np.long)
        # event2 = event[(event[:, 0] >= ts[id_start+1]-dt) & (event[:, 0] <= ts[id_end]+dt), :].astype(np.long)
        # event_vox0 = self.event2vox(event0.copy())
        # event_vox1 = self.event2vox(event1.copy())
        # event_vox2 = self.event2vox(event2.copy())
        # event_vox = torch.stack([event_vox0, event_vox1, event_vox2], dim=0)

        if event01.shape[0] == 0:
            H, W = self.dim
            event01 = torch.zeros(self.nb_of_bin, H, W, dtype=torch.float32, device='cpu')
        else:
            event01 = self.event2vox(event01)

        if event12.shape[0] == 0:
            H, W = self.dim
            event21 = torch.zeros(self.nb_of_bin, H, W, dtype=torch.float32, device='cpu')
        else:
            event21 = self.reverse(event12)
            event21 = self.event2vox(event21)

        return event01, event21, self.img_file[idx]
        # return event_vox, self.img_file[idx]


    def __len__(self):
        return len(self.gt_file)

    def reverse(self, event):
        if len(event) == 0:
            return
        event[:, 0] = (event[-1,0] - event[:, 0])
        event[:, -1] = -event[:, -1]
        event = np.copy(np.flipud(event))
        return event

    def event2vox(self, event):
        event = torch.from_numpy(event).float()
        H, W = self.dim

        voxel_grid = torch.zeros(self.nb_of_bin, H, W, dtype=torch.float32, device='cpu')
        vox = voxel_grid.ravel()

        t, x, y, p = event.t()
        t = t.long()
        time_max = t.max()
        time_min = t.min()

        t = (t-time_min) * (self.nb_of_bin - 1) / (time_max-time_min)
        t = t.float()
        left_t, right_t = t.floor(), t.floor()+1
        left_x, right_x = x.float().floor(), x.float().floor()+1
        left_y, right_y = y.float().floor(), y.float().floor()+1

        for lim_x in [left_x, right_x]:
            for lim_y in [left_y, right_y]:
                for lim_t in [left_t, right_t]:
                    mask = (0 <= lim_x) & (0 <= lim_y) & (0 <= lim_t) & (lim_x <= W-1) & (lim_y <= H-1) & (lim_t <= self.nb_of_bin-1)
                    lin_idx = lim_x.long() + lim_y.long() * W + lim_t.long() * W * H
                    weight = p * (1-(lim_x-x).abs()) * (1-(lim_y-y).abs()) * (1-(lim_t-t).abs())
                    vox.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())

        return voxel_grid

    def data_augmentation(self, I0, I1, I2, event01, event21, crop_size=(256, 256), train=True):
        if train:
            args = transforms.Compose([transforms.RandomCrop(crop_size), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
        else:
            args = transforms.Compose([transforms.CenterCrop(crop_size)])
        x = torch.cat([I0, I1, I2, event01, event21], dim=0)
        x = args(x)
        I0 = x[:3, ...]
        I1 = x[3:6, ...]
        I2 = x[6:9, ...]
        event01 = x[9:9+self.nb_of_bin, ...]
        event21 = x[9+self.nb_of_bin:, ...]
        return I0, I1, I2, event01, event21


class dataset_efsai_data(data.Dataset):
    def __init__(self, train=True, arg=False):
        if train:
            self.file = '/home/lisiqi/data/EF-SAI-Dataset/train.txt'
        else:
            self.file = '/home/lisiqi/data/EF-SAI-Dataset/test.txt'

        self.event_file, self.gt_file, self.img_file = [], [], []
        self.dim = (260, 346)
        self.nb_of_bin = 36

        with open(self.file, 'r') as f:
            for line in f.readlines():
                p1 = line.strip('\n')
                self.event_file.append(p1.split(' ')[0].replace('_refocus', ''))
                self.img_file.append(p1.split(' ')[1].replace('_refocus', ''))
                self.gt_file.append(p1.split(' ')[2])

        self.train = (train & arg)
        self.arg = arg

    def __getitem__(self, idx):
        FrameData = np.load(self.img_file[idx], allow_pickle=True).item()
        EventData = np.load(self.event_file[idx], allow_pickle=True).item()
        img = FrameData['images']
        # img1 = img[mid-8:mid+9, ...]
        img = img.astype(np.float)/255

        ts = FrameData['time_stamp']
        t_ref = FrameData['ref_t']
        t_ref -= ts
        mid = np.where(t_ref == 0)[0][0]

        t0 = ts.min()
        ts -= t0
        ts *= 1e6
        dt = 10000

        id_start = mid-8
        id_end = mid+9

        event = np.stack([EventData['events']['t'].astype(np.float64), EventData['events']['x'].astype(np.float64),
                          EventData['events']['y'].astype(np.float64), EventData['events']['p'].astype(np.float64)]).T
        event[:, 0] -= t0
        event[:, 0] *= 1e6

        event0 = event[(event[:, 0] >= ts[id_start-1]-dt) & (event[:, 0] <= ts[id_end-2]+dt), :].astype(np.long)
        event1 = event[(event[:, 0] >= ts[id_start]-dt) & (event[:, 0] <= ts[id_end-1]+dt), :].astype(np.long)
        event2 = event[(event[:, 0] >= ts[id_start+1]-dt) & (event[:, 0] <= ts[id_end]+dt), :].astype(np.long)
        event_vox0 = self.event2vox(event0.copy())
        event_vox1 = self.event2vox(event1.copy())
        event_vox2 = self.event2vox(event2.copy())
        event_vox = torch.stack([event_vox0, event_vox1, event_vox2], dim=0)

        imgs_0 = img[id_start-1:id_end-1]
        imgs_1 = img[id_start:id_end]
        imgs_2 = img[id_start+1:id_end+1]
        imgs = np.stack([imgs_0, imgs_1, imgs_2], axis=0)

        mask = []
        for k in range(id_start-1, id_end+1):
            e1 = event[(event[:,0]>ts[k]-dt)&(event[:,0]<ts[k]+dt), :]
            ecm = self.event2ecm(e1)
            m = (ecm!=0)
            mask.append(m)
        mask0 = 1 - np.concatenate(mask[:-2], axis=0).astype(np.float)
        mask1 = 1 - np.concatenate(mask[1:-1], axis=0).astype(np.float)
        mask2 = 1 - np.concatenate(mask[2:], axis=0).astype(np.float)
        mask = torch.stack([torch.from_numpy(mask0), torch.from_numpy(mask1), torch.from_numpy(mask2)], dim=0)

        return event_vox, imgs, mask, self.img_file[idx]

    def __len__(self):
        return len(self.gt_file)

    def reverse(self, event):
        if len(event) == 0:
            return
        event[:, 0] = (event[-1,0] - event[:, 0])
        event[:, -1] = -event[:, -1]
        event = np.copy(np.flipud(event))
        return event

    def event2vox(self, event):
        event = torch.from_numpy(event).float()
        H, W = self.dim

        voxel_grid = torch.zeros(self.nb_of_bin, H, W, dtype=torch.float32, device='cpu')
        vox = voxel_grid.ravel()

        t, x, y, p = event.t()
        t = t.long()
        time_max = t.max()
        time_min = t.min()

        t = (t-time_min) * (self.nb_of_bin - 1) / (time_max-time_min)
        t = t.float()
        left_t, right_t = t.floor(), t.floor()+1
        left_x, right_x = x.float().floor(), x.float().floor()+1
        left_y, right_y = y.float().floor(), y.float().floor()+1

        for lim_x in [left_x, right_x]:
            for lim_y in [left_y, right_y]:
                for lim_t in [left_t, right_t]:
                    mask = (0 <= lim_x) & (0 <= lim_y) & (0 <= lim_t) & (lim_x <= W-1) & (lim_y <= H-1) & (lim_t <= self.nb_of_bin-1)
                    lin_idx = lim_x.long() + lim_y.long() * W + lim_t.long() * W * H
                    weight = p * (1-(lim_x-x).abs()) * (1-(lim_y-y).abs()) * (1-(lim_t-t).abs())
                    vox.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())

        return voxel_grid

    def event2ecm(self, event):
        event = torch.from_numpy(event).float()
        H, W = self.dim

        voxel_grid = torch.zeros(1, H, W, dtype=torch.float32, device='cpu')
        vox = voxel_grid.ravel()

        t, x, y, p = event.t()
        left_x, right_x = x.float().floor(), x.float().floor()+1
        left_y, right_y = y.float().floor(), y.float().floor()+1

        for lim_x in [left_x, right_x]:
            for lim_y in [left_y, right_y]:
                    mask = (0 <= lim_x) & (0 <= lim_y) & (lim_x <= W-1) & (lim_y <= H-1)
                    lin_idx = lim_x.long() + lim_y.long() * W
                    weight = (1 - (lim_x - x).abs()) * (1 - (lim_y - y).abs())
                    vox.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())

        return voxel_grid

    def data_augmentation(self, I0, I1, I2, event01, event21, crop_size=(256, 256), train=True):
        if train:
            args = transforms.Compose([transforms.RandomCrop(crop_size), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
        else:
            args = transforms.Compose([transforms.CenterCrop(crop_size)])
        x = torch.cat([I0, I1, I2, event01, event21], dim=0)
        x = args(x)
        I0 = x[:3, ...]
        I1 = x[3:6, ...]
        I2 = x[6:9, ...]
        event01 = x[9:9+self.nb_of_bin, ...]
        event21 = x[9+self.nb_of_bin:, ...]
        return I0, I1, I2, event01, event21


class dataset_occ400_data(data.Dataset):
    def __init__(self, train=True, arg=False):
        if train:
            self.file = '/home/lisiqi/data/EF-SAI-Dataset/train.txt'
        else:
            self.file = '/home/lisiqi/data/EF-SAI-Dataset/test.txt'

        self.event_file, self.gt_file, self.img_file = [], [], []
        self.dim = (260, 346)
        self.nb_of_bin = 36

        with open(self.file, 'r') as f:
            for line in f.readlines():
                p1 = line.strip('\n')
                self.event_file.append(p1.split(' ')[0].replace('_refocus', ''))
                self.img_file.append(p1.split(' ')[1].replace('_refocus', ''))
                self.gt_file.append(p1.split(' ')[2])

        self.train = (train & arg)
        self.arg = arg

    def __getitem__(self, idx):
        FrameData = np.load(self.img_file[idx], allow_pickle=True).item()
        EventData = np.load(self.event_file[idx], allow_pickle=True).item()
        img = FrameData['images']
        # img1 = img[mid-8:mid+9, ...]
        img = img.astype(np.float)/255

        ts = FrameData['time_stamp']
        t_ref = FrameData['ref_t']
        t_ref -= ts
        mid = np.where(t_ref == 0)[0][0]

        t0 = ts.min()
        ts -= t0
        ts *= 1e6
        dt = 10000

        id_start = mid-8
        id_end = mid+9

        event = np.stack([EventData['events']['t'].astype(np.float64), EventData['events']['x'].astype(np.float64),
                          EventData['events']['y'].astype(np.float64), EventData['events']['p'].astype(np.float64)]).T
        event[:, 0] -= t0
        event[:, 0] *= 1e6

        event0 = event[(event[:, 0] >= ts[id_start-1]-dt) & (event[:, 0] <= ts[id_end-2]+dt), :].astype(np.long)
        event1 = event[(event[:, 0] >= ts[id_start]-dt) & (event[:, 0] <= ts[id_end-1]+dt), :].astype(np.long)
        event2 = event[(event[:, 0] >= ts[id_start+1]-dt) & (event[:, 0] <= ts[id_end]+dt), :].astype(np.long)
        event_vox0 = self.event2vox(event0.copy())
        event_vox1 = self.event2vox(event1.copy())
        event_vox2 = self.event2vox(event2.copy())
        event_vox = torch.stack([event_vox0, event_vox1, event_vox2], dim=0)

        imgs_0 = img[id_start-1:id_end-1]
        imgs_1 = img[id_start:id_end]
        imgs_2 = img[id_start+1:id_end+1]
        imgs = np.stack([imgs_0, imgs_1, imgs_2], axis=0)

        mask = []
        for k in range(id_start-1, id_end+1):
            e1 = event[(event[:,0]>ts[k]-dt)&(event[:,0]<ts[k]+dt), :]
            ecm = self.event2ecm(e1)
            m = (ecm!=0)
            mask.append(m)
        mask0 = 1 - np.concatenate(mask[:-2], axis=0).astype(np.float)
        mask1 = 1 - np.concatenate(mask[1:-1], axis=0).astype(np.float)
        mask2 = 1 - np.concatenate(mask[2:], axis=0).astype(np.float)
        mask = torch.stack([torch.from_numpy(mask0), torch.from_numpy(mask1), torch.from_numpy(mask2)], dim=0)

        return event_vox, imgs, mask, self.img_file[idx]

    def __len__(self):
        return len(self.gt_file)

    def reverse(self, event):
        if len(event) == 0:
            return
        event[:, 0] = (event[-1,0] - event[:, 0])
        event[:, -1] = -event[:, -1]
        event = np.copy(np.flipud(event))
        return event

    def event2vox(self, event):
        event = torch.from_numpy(event).float()
        H, W = self.dim

        voxel_grid = torch.zeros(self.nb_of_bin, H, W, dtype=torch.float32, device='cpu')
        vox = voxel_grid.ravel()

        t, x, y, p = event.t()
        t = t.long()
        time_max = t.max()
        time_min = t.min()

        t = (t-time_min) * (self.nb_of_bin - 1) / (time_max-time_min)
        t = t.float()
        left_t, right_t = t.floor(), t.floor()+1
        left_x, right_x = x.float().floor(), x.float().floor()+1
        left_y, right_y = y.float().floor(), y.float().floor()+1

        for lim_x in [left_x, right_x]:
            for lim_y in [left_y, right_y]:
                for lim_t in [left_t, right_t]:
                    mask = (0 <= lim_x) & (0 <= lim_y) & (0 <= lim_t) & (lim_x <= W-1) & (lim_y <= H-1) & (lim_t <= self.nb_of_bin-1)
                    lin_idx = lim_x.long() + lim_y.long() * W + lim_t.long() * W * H
                    weight = p * (1-(lim_x-x).abs()) * (1-(lim_y-y).abs()) * (1-(lim_t-t).abs())
                    vox.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())

        return voxel_grid

    def event2ecm(self, event):
        event = torch.from_numpy(event).float()
        H, W = self.dim

        voxel_grid = torch.zeros(1, H, W, dtype=torch.float32, device='cpu')
        vox = voxel_grid.ravel()

        t, x, y, p = event.t()
        left_x, right_x = x.float().floor(), x.float().floor()+1
        left_y, right_y = y.float().floor(), y.float().floor()+1

        for lim_x in [left_x, right_x]:
            for lim_y in [left_y, right_y]:
                    mask = (0 <= lim_x) & (0 <= lim_y) & (lim_x <= W-1) & (lim_y <= H-1)
                    lin_idx = lim_x.long() + lim_y.long() * W
                    weight = (1 - (lim_x - x).abs()) * (1 - (lim_y - y).abs())
                    vox.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())

        return voxel_grid

    def data_augmentation(self, I0, I1, I2, event01, event21, crop_size=(256, 256), train=True):
        if train:
            args = transforms.Compose([transforms.RandomCrop(crop_size), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
        else:
            args = transforms.Compose([transforms.CenterCrop(crop_size)])
        x = torch.cat([I0, I1, I2, event01, event21], dim=0)
        x = args(x)
        I0 = x[:3, ...]
        I1 = x[3:6, ...]
        I2 = x[6:9, ...]
        event01 = x[9:9+self.nb_of_bin, ...]
        event21 = x[9+self.nb_of_bin:, ...]
        return I0, I1, I2, event01, event21


if __name__ == '__main__':
    import cv2
    from torch.nn import functional as F
    from PIL import Image
    # d = dataset_efsai_flow(train=True)
    d = dataset_efsai_data(train=False)
    for k in range(len(d)):
        # event01, event21, folder = d.__getitem__(k)
        # f1, idx = os.path.split(folder)
        # x = torch.cat([event01, event21], dim=0)
        # np.save(os.path.join(os.path.split(f1)[0], 'data', f'{idx.strip(".npy")}_flow.npy'), np.array(x))
        # print(k)

        event_vox, imgs, mask, folder = d.__getitem__(k)
        f1, idx = os.path.split(folder)
        np.save(os.path.join(os.path.split(f1)[0], 'data', f'{idx.strip(".npy")}_event3.npy'), np.array(event_vox))
        np.save(os.path.join(os.path.split(f1)[0], 'data', f'{idx.strip(".npy")}_img3.npy'), np.array(imgs))
        np.save(os.path.join(os.path.split(f1)[0], 'data', f'{idx.strip(".npy")}_mask3.npy'), np.array(mask))
        print(k)

        # break

    # os.makedirs('./check', exist_ok=True)
    # transforms.ToPILImage()(I0).save('./check/0_Img.png')
    # transforms.ToPILImage()(I1).save('./check/1_Img.png')
    # transforms.ToPILImage()(I2).save('./check/2_Img.png')
    # for k in range(5):
    #     transforms.ToPILImage()(event01[k]).save(f'./check/e01_{k}.png')
    # for k in range(5):
    #     transforms.ToPILImage()(event21[k]).save(f'./check/e21_{k}.png')