import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms


from PIL import Image
from torch.utils.data import Dataset

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


mean = torch.tensor(MEAN).cuda().view(1,3,1,1) * 255
std = torch.tensor(STD).cuda().view(1,3,1,1) * 255


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] = torch.from_numpy(nump_array)

    return tensor, targets


class PrefetchedWrapper(object):
    def prefetched_loader(loader):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda()
                next_target = next_target.cuda()
                next_input = next_input.float()
                next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.epoch = 0

    def __iter__(self):
        if (self.dataloader.sampler is not None and
            isinstance(self.dataloader.sampler,
                       torch.utils.data.distributed.DistributedSampler)):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.dataloader)

def get_pytorch_train_loader(data_path, batch_size, custom_oi_path='', oi_thresh=0.4, workers=5, _worker_init_fn=None, input_size=224):
    traindir = os.path.join(data_path, 'train')
    train_transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                ])
    train_dataset = datasets.ImageFolder(
            traindir, train_transform)
    if len(custom_oi_path):
        root_dir = os.path.dirname(custom_oi_path)
        train_dataset = JoinedDataset(train_dataset, CustomOI(root_dir, custom_oi_path, transform=train_transform, threshold=oi_thresh))

    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate)

    return PrefetchedWrapper(train_loader), len(train_loader)

def get_pytorch_val_loader(data_path, batch_size, workers=5, _worker_init_fn=None, input_size=224):
    valdir = os.path.join(data_path, 'val')
    val_dataset = datasets.ImageFolder(
            valdir, transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                ]))

    if torch.distributed.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True,
            collate_fn=fast_collate)

    return PrefetchedWrapper(val_loader), len(val_loader)


from torch.utils.data import Dataset
import json

class CustomOI(Dataset):
    def __init__(self, image_root_dir, dump_path, threshold=0.4, transform=None):
        self.image_root_dir = image_root_dir
        self.transform = transform
        self.images_info = []
        with open(dump_path, 'r') as f:
            img_data = json.load(f)

            for item in img_data:
                name, label, conf = item
                if conf > threshold:
                    self.images_info.append((os.path.join(self.image_root_dir, name), label))


    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.images_info[idx][0])
            label = self.images_info[idx][1]
            if len(img.mode) != 3: # we need to convert gray and rgba images to rgb
                rgbimg = Image.new('RGB', img.size)
                rgbimg.paste(img)
                img = rgbimg
        except:
            print('Unable to read the image: ' + self.images_info[idx][0])
            img = torch.zeros((3, 224, 224))

        if self.transform:
            img = self.transform(img)
        sample = (img, label)
        return sample


class JoinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx):
        if idx >= len(self.dataset1):
            idx -= len(self.dataset1)
            return self.dataset2[idx]
        else:
            return self.dataset1[idx]