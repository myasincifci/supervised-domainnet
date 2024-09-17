import os
import torch
from torch.utils.data import random_split, Subset
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as T
from torchvision.transforms import functional as TF
import pytorch_lightning as pl
from supervised_domainnet.datasets.domainnet_dataset import DomainNetDataset
from yasin_utils.data import TransformWrapper
from yasin_utils.image import imagenet_normalize

class DomainNetDM(pl.LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg

        self.train_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            imagenet_normalize
        ])

        self.val_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            imagenet_normalize
        ])

        self.dataset = DomainNetDataset(
            root=os.path.join(self.cfg.data.path, 'domainnet_v1.0'), 
        )

        self.num_classes = self.dataset.num_classes

    def setup(self, stage: str) -> None:        
        if stage == 'fit':
            train_set, val_set = random_split(self.dataset, lengths=(0.9,0.1))
            self.train_set = TransformWrapper(train_set, self.train_transform)
            self.val_set = TransformWrapper(val_set, self.val_transform)
            
        elif stage == 'test':
            pass
        
        elif stage == 'predict':
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.param.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:   
        return DataLoader(
            self.val_set,
            batch_size=self.cfg.param.batch_size,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
def main():
    cfg = {
        'data': {
            'name': 'domainnet',
            'path': '/data',
            'num_workers': 0
        }
    } 
    cfg = DictConfig(cfg)

    dm = DomainNetDM(cfg)

    dm.setup(stage='fit')

    a=1

if __name__ == '__main__':
    from omegaconf import DictConfig

    main()