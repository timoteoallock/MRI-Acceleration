
import pathlib
import pytorch_lightning as pl

from fastmri.data.subsample import create_mask_for_mask_type

from twod import KspaceDataTransform, FastMriDataModule,FastMRIRawDataSample, CustomUnetDataTransform, CustomSliceDataset, MriModule, UnetModule
from pytorch_lightning.strategies.ddp import DDPStrategy

#define mask 
mask = create_mask_for_mask_type(
        'equispaced', [0.08], [4]    )

# train_transform = KspaceDataTransform('lightning','complex', 'poisson','singlecoil',  mask_func=mask)
# val_transform = KspaceDataTransform('lightning','complex', 'poisson','singlecoil', mask_func=mask)
# test_transform = KspaceDataTransform('lightning','complex', 'poisson','singlecoil')
train_transform = KspaceDataTransform('lightning', 'complex','equispaced','singlecoil', mask_func = mask, use_seed = False)
val_transform = KspaceDataTransform('lightning', 'complex', 'equispaced',  'singlecoil', mask_func = mask)
test_transform = KspaceDataTransform('lightning', 'complex', 'equispaced', 'singlecoil', mask_func = mask)
data_path = pathlib.Path("/home/changchen/dataset/mri_data")

# define datamodule, I found that using 15 workers and 40 batch_size doesn't overwhelm 
# the GPU memory, the max batch was 60 before running out of memory. To use higher batch
# sizes decrease the number of workers. 
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
        approach = 'complex', ## this MUST match the approach in the transform
        in_chans=2,  # number of input channels to U-Net (1 for real and 2 for kspace and complex)
        out_chans=2,  # number of output chanenls to U-Net
        chans=90,  # number of top-level U-Net channels
        num_pool_layers=4,  # number of U-Net pooling layers
        drop_prob=0.0,  # dropout probability
        lr=0.001,  # RMSProp learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate (lr*lr_gamma)
        weight_decay=0.00, 
        # weight decay regularization strength
    )
# checkpoint_path = my_path
# model = model.load_from_checkpoint(checkpoint_path)

from pytorch_lightning.loggers import TensorBoardLogger 



# Initialize the TensorBoardLogger
logger = TensorBoardLogger("logss", name="lightning_images")
## to launch tensorboard from terminal use 
## tensorboard --logdir ... and put in place of ... the location of the log folder
# Initialize the Trainer with the logger


if __name__ == "__main__":
    # Initialize the Trainer with DDP and the logger
    trainer = pl.Trainer(accelerator='gpu', devices=1,max_epochs = 60, strategy = DDPStrategy(find_unused_parameters=False),logger = logger)
    trainer.fit(model, datamodule=data_module)