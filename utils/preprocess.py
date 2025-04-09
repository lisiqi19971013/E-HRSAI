import glob
from PIL import Image
from torch.utils import data
import os
import torch
import numpy as np
import cv2
from torchvision import transforms


class dataset(data.Dataset):
    def __init__(self, file, nb_of_bin=36, nb_of_frame=12):
        if 'train' not in file and 'test' not in file:
            raise ValueError

        self.file = file
        self.event_file, self.gt_file, self.img_folder = [], [], []
        self.dim = (260, 346)
        self.nb_of_bin = nb_of_bin
        self.nb_of_frame = nb_of_frame
        with open(file, 'r') as f:
            for line in f.readlines():
                p1 = line.strip('\n').split(' ')[0]
                self.event_file.append(p1)
                self.gt_file.append(os.path.join(os.path.split(p1)[0], 'image.jpg'))
                self.img_folder.append(os.path.split(p1)[0])

    def find_nearest(self, ts, ts_list):
        ts_list = np.array(ts_list)
        dt = ts_list - ts
        dt = np.abs(dt)
        idx = np.where(dt == dt.min())
        return idx[0][0]

    def __getitem__(self, idx):
        event = np.load(self.event_file[idx])
        folder = self.img_folder[idx]

        # event_vox = torch.from_numpy(np.load(os.path.join(folder, 'event_vox.npy')))
        event_grid = self.event_grid(event)
        event_vox = self.event2vox(event)
        return event_vox, event_grid, folder

    def __len__(self):
        return len(self.gt_file)

    def event2vox(self, event):
        event = torch.from_numpy(event).float()
        H, W = self.dim

        voxel_grid = torch.zeros(self.nb_of_bin, H, W, dtype=torch.float32, device='cpu')
        vox = voxel_grid.ravel()

        t, p, x, y = event.t()
        if p.min() == 0:
            p = p * 2 - 1
        # print(p.max(), p.min())
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

        t, p, x, y = event.t()
        left_x, right_x = x.float().floor(), x.float().floor()+1
        left_y, right_y = y.float().floor(), y.float().floor()+1

        for lim_x in [left_x, right_x]:
            for lim_y in [left_y, right_y]:
                    mask = (0 <= lim_x) & (0 <= lim_y) & (lim_x <= W-1) & (lim_y <= H-1)
                    lin_idx = lim_x.long() + lim_y.long() * W
                    weight = (1 - (lim_x - x).abs()) * (1 - (lim_y - y).abs())
                    vox.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())

        return voxel_grid

    def visEvent(self, events, nb_of_bin, folder, format="%04d.jpg"):
        os.makedirs(folder, exist_ok=True)
        dt = (events[:, 0].max()-events[:, 0].min()) / nb_of_bin
        for k in range(nb_of_bin):
            e1 = events[(events[:, 0] >= k * dt) & (events[:, 0] <= (k + 1) * dt), :]
            ecm = np.zeros([260, 346, 3])
            for e in e1:
                if e[1] == 1:
                    ecm[int(e[3]), int(e[2]), 2] += 1
                else:
                    ecm[int(e[3]), int(e[2]), 0] += 1
            ecm[ecm > 0.8 * ecm.max()] = 0.8 * ecm.max()
            ecm /= ecm.max()
            ecm *= 255
            cv2.imwrite(os.path.join(folder, format % k), ecm)

    def event_grid(self, events):
        events = torch.from_numpy(events)

        t, p, x, y = events.t()
        if min(t.shape) == 0:
            print("Warning")

        t -= t.min()
        time_max = t.max()

        num_voxels = int(2 * np.prod(self.dim) * self.nb_of_bin)
        vox = events[0].new_full([num_voxels, ], fill_value=0)
        H, W = self.dim
        C = self.nb_of_bin

        # normalizing timestamps

        t = t * C/(time_max+1)
        t = t.long()

        # bin = 1 / C
        # s_bin = 0
        # e_bin = bin
        # for i in range(C):
        #     t[(s_bin <= t) == (t < e_bin)] = i
        #     s_bin += bin
        #     e_bin += bin
        # t[-1] = C - 1

        idx = x + W * y + W * H * t + W * H * C * p
        values = torch.full_like(t, 1)

        # draw in voxel grid
        vox.put_(idx.long(), values, accumulate=True)

        vox = vox.view(2, C, H, W)
        vox = torch.cat([vox[0, ...], vox[1, ...]], 0)
        # vox = vox.numpy()
        return vox


def visImage(img, folder, name):
    img = np.array(img)
    H, W = img.shape[1], img.shape[2]
    full_img = np.zeros([H*3, W*4, 3])
    for i in range(3):
        for j in range(4):
            if i == 2 and j == 3:
                break
            idx = i * 4 + j
            img1 = img[idx*3:idx*3+3, :, :].transpose([1,2,0])
            full_img[H*i:H*i+H, W*j:W*j+W, :] = img1
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, name), full_img)


def visEvent(event_vox, folder, name):
    img = np.array(event_vox)
    H, W = event_vox.shape[1], event_vox.shape[2]
    full_img = np.zeros([H*8, W*9])
    for i in range(8):
        for j in range(9):
            idx = i * 9 + j
            img1 = img[idx, :, :]
            img1 = img1/img1.max()*255
            full_img[H*i:H*i+H, W*j:W*j+W] = img1
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, name), full_img)


if __name__ == '__main__':
    file = '/home/lisiqi/data/SAIdata/test.txt'
    d = dataset(file)
    os.makedirs('./check1', exist_ok=True)

    # for k in range(len(d)):
    #     event_vox, event_grid, folder = d.__getitem__(k)
    #     # e = np.load(d.event_file[k])
    #     # l = []
    #     # for j in range(72):
    #     #     l.append(event_vox[j].max())
    #     # print(k, min(l), max(l))
    #
    #     # visImage(img, './check1/', '%d_img.jpg'%k)
    #     # visImage(mask*255, './check1/', '%d_mask.jpg'%k)
    #     # visEvent(event_vox, './check1/', '%d_event.jpg'%k)
    #     # cv2.imwrite('./check1/%d_gt.jpg'%k, np.array(gt_img).transpose([1,2,0]))
    #     # cv2.imwrite('./check1/%d_gt1.jpg'%k, np.array(img[15:18, ...]).transpose([1,2,0]))
    #
    #     np.save(os.path.join(folder, 'event_vox1.npy'), np.array(event_vox))
    #     np.save(os.path.join(folder, 'event_grid1.npy'), np.array(event_grid).astype(np.uint8))
    #     print(k, event_vox.max(), event_vox.min(), event_grid.max(), event_grid.min())
    #     # break



    # # event = d.__getitem__(10)
    # # ecm = d.event2ecm(event)
    # # os.makedirs('./check', exist_ok=True)
    # # for k in range(72):
    # #     img1 = event_vox[k]/event_vox[k].max()*255
    # #     cv2.imwrite('./check/%d_vox.jpg'%k, np.array(img1))
    # # for k in range(12):
    # #     img1 = img[k*3:k*3+3, ...].transpose([1,2,0])
    # #     m = mask[k*3, ...] * 255
    # #     cv2.imwrite('./check/%d_img.jpg'%k, np.array(img1))
    # #     cv2.imwrite('./check/%d_mask.jpg'%k, np.array(m))
    # # cv2.imwrite('./check/gt.jpg', gt_img.transpose([1,2,0]))