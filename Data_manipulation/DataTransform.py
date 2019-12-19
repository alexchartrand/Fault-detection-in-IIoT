import torch

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, fault, minMax = sample['data'], sample['fault'], sample['minMax']
        sample['data'] = torch.from_numpy(data).float()
        sample['fault'] = torch.from_numpy(fault).float()
        sample['minMax'] = torch.from_numpy(minMax).float()
        return sample

class ToNumpy(object):
    """Convert torch.tensor to ndArray."""

    def __call__(self, sample):
        data, fault, minMax = sample['data'], sample['fault'], sample['minMax']
        sample['data'] = data.cpu().detach().numpy()
        sample['fault'] = fault.cpu().detach().numpy()
        sample['minMax'] = minMax.cpu().detach().numpy()
        return sample

class To3DTimeSeries(object):
    """Convert to 3D timeseries"""

    def __call__(self, sample):
        data = sample['data']
        sample['data'] = torch.unsqueeze(data, 2)
        return sample