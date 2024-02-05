
import pathlib
import pytorch_lightning as pl

from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform

from utils import KspaceDataTransform, FastMriDataModule,FastMRIRawDataSample, CustomUnetDataTransform, CustomSliceDataset, MriModule, UnetModule
from pytorch_lightning.strategies.ddp import DDPStrategy
mask = create_mask_for_mask_type(
        'equispaced', [0.08], [8]    )

train_transform = UnetDataTransform('singlecoil', mask_func=mask, use_seed=False)
val_transform = UnetDataTransform('singlecoil', mask_func=mask)
test_transform = UnetDataTransform('singlecoil')
# train_transform = KspaceDataTransform('lightning', 'image', 'singlecoil', mask_func = mask, use_seed = False)
# val_transform = KspaceDataTransform('lightning', 'image', 'singlecoil', mask_func = mask)
# test_transform = KspaceDataTransform('lightning', 'image', 'singlecoil', mask_func = mask)
data_path = pathlib.Path("/home/changchen/dataset/mri_data")

data_module = FastMriDataModule(
        data_path=data_path,
        challenge='singlecoil',
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        combine_train_val=False,
        num_workers=15,
        batch_size= 40,
        distributed_sampler=True          
    )


model = UnetModule(
    'real',
        in_chans=1,  # number of input channels to U-Net
        out_chans=1,  # number of output chanenls to U-Net
        chans=90,  # number of top-level U-Net channels
        num_pool_layers=4,  # number of U-Net pooling layers
        drop_prob=0.0,  # dropout probability
        lr=0.001,  # RMSProp learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.00, 
        # weight decay regularization strength
    )
#checkpoint_path = '/home/changchen/logss/lightning_images/version_1/checkpoints/epoch=29-step=26070.ckpt'

from pytorch_lightning.loggers import TensorBoardLogger ## TEST THIS 

# Create your LightningModule and DataLoader

# Initialize the TensorBoardLogger
logger = TensorBoardLogger("logss", name="lightning_images")

# Initialize the Trainer with the logger


if __name__ == "__main__":
    # Initialize the Trainer with DDP and the logger
    trainer = pl.Trainer(accelerator='gpu', devices=1,max_epochs = 60, strategy = DDPStrategy(find_unused_parameters=False),logger = logger)
    #tainer = pl.Trainer(strategy="ddp_find_unused_parameters_false", accelerator="gpu", devices="auto")

    trainer.fit(model, datamodule=data_module)
    #, ckpt_path = checkpoint_path
