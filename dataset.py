from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from pathlib import Path
import struct
from torch.utils.data import DataLoader
import glob

class NerfDataset(Dataset):
    def __init__(self, datapath, mode, **kwargs):
        super(NerfDataset, self).__init__()
        self.datapath = datapath
        self.mode = mode
        assert self.mode in ["train", "val", "test"]
        self.listfile = glob.glob(self.datapath + 'cams/*_cam.txt')

    def __len__(self):
        return len(self.listfile)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

        return intrinsics, extrinsics

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def __getitem__(self, idx):
        img_filename = os.path.join(self.datapath, 'blended_images/{}.jpg'.format(str(idx).zfill(8)))
        proj_mat_filename = os.path.join(self.datapath, 'cams/{}_cam.txt'.format(str(idx).zfill(8)))

        img = self.read_img(img_filename)
        K, Rt = self.read_cam_file(proj_mat_filename)

        return {
            "img": img,
            "K": K,
            "Rt": Rt
        }


if __name__ == "__main__":

    dataset = NerfDataset("../blended_mvs/5a3ca9cb270f0e3f14d0eddb/", 'train')

    item = dataset[50]
    print(item.keys(), len(dataset))
    print("img", item["img"].shape)
    print("K", item["K"].shape)
    print("Rt", item["Rt"].shape)

    minx = 0
    miny = 0
    minz = 0

    for i in range(64):
        item = dataset[i]
        if minx < item["Rt"][0,3]:
            minx = item["Rt"][0,3]
        if miny < item["Rt"][1,3]:
            miny = item["Rt"][1,3]
        if minz < item["Rt"][2,3]:
            minz = item["Rt"][2,3]

    loader = DataLoader(dataset, 1, shuffle=True, num_workers=8, drop_last=True)

    import numpy as np
    from visualizer import CameraPoseVisualizer

    n = max(minx, miny, minz)

    # argument : the minimum/maximum value of x, y, z
    visualizer = CameraPoseVisualizer([-n, n], [-n, n], [0, n])

    for batch_idx, sample in enumerate(loader):
        item = sample
        # print("img", item["img"].shape)
        # print("K", item["K"].shape)
        # print("Rt", item["Rt"].shape)
        # break
        # print(item["Rt"].squeeze(0).detach().cpu().numpy().shape)
        visualizer.extrinsic2pyramid(item["Rt"].squeeze(0).detach().cpu().numpy(), 'c', 0.1)

    visualizer.show()
    visualizer.save()
