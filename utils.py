import logging
import os 
import pickle
import random
import h5py
import numpy as np
import fastmri
from pathlib import Path
import fastmri.data 
import fastmri.data.transforms as T 
from argparse import ArgumentParser
import pytorch_lightning as pl
import fastmri
from fastmri.data import CombinedSliceDataset, SliceDataset
import fastmri.data.transforms
from fastmri.data.transforms import UnetDataTransform
import torch
from typing import (
    Any,
    Callable,
    Dict,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
"""
To adapt those classes to your specific task refer to 
https://github.com/facebookresearch/fastMRI
"""

"""
This file contains the classes that can be used to train 2D input channel Unets
If the goal is to train on the real image, one can either use the default classes of FastMRI
Library combined with some functions of this file, as done in the tutorial, 
or use only the functions in this file. (by inputting 'real' as the approach in transform)
For 2D channel training, this file contains all the functions needed for 
the task. 
The important classes in this file are: 
    1) KspaceDataTransform: combined with FastMriDataModule returns data 
                            in complex forms and serves the same purpose of 
                            UnetDataTransform from fastmri.data.transforms
    2) CustomSliceDataset: serves the same purpose of SliceDataset and in addition 
                            supports as transforms KspaceDataTransform
    3) MRIModule: parent class of UnetModule that was modified so that now it
                  displays in Tensorboard the input image as well
    4) UnetModule: same as UnetModule from fastmri library which has been modified to 
                   support 2 input channel training
    
"""

class KspaceDataTransform(UnetDataTransform) :
    
    """
    Extra Args:
        platform: either "pytorch" or "lightning", the returned data 
            is suited to the selected platform. By selecting pytormasked_kspace = kspace_torch*torch.from_numpy(self.mask_func).unsqueeze(-1)+ 0.0h the data 
            can be put into a torch.utils.data.DataLoader, refer to the tutorial 
            for "lightning".
        objective: either "kspace" or "image", returns the masked kspace and target
            kspace or the masked image and the target image. 
            Each datapoint has dimension [2,320,320]
        mask_type : only important if you use poisson mask, but even if not , specify the type
    """
    def __init__(self, platform, objective, mask_type, *args, **kwargs): 
        super().__init__(*args,**kwargs)
        if platform not in ("pytorch", "lightning"): 
            raise ValueError('platform should be either "pytorch" or "lightning"')
        self.platform = platform
        if objective not in ("kspace", "complex", "real"): 
            raise ValueError('objective should be either "kspace" , "real" or "complex"')
        self.objective = objective 
        self.mask_type = mask_type
    
    def __call__(self, kspace ,mask, target, attrs, fname, slice_num):
        kspace_torch = T.to_tensor(kspace)
        if self.mask_type == 'poisson':
            masked_kspace = kspace_torch*torch.from_numpy(self.mask_func).unsqueeze(-1)+ 0.0
            masked_kspace = masked_kspace.float()
        else: 
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace = T.apply_mask(kspace_torch, self.mask_func, seed = seed)[0]
        # modulus_phase_tensor = Modstand*np.exp(1j*Phase)
        max_value  = attrs['max'] if 'max' in attrs.keys() else 0.0
   
            
            # masked_kspace1, mean1, std1 = T.normalize_instance(masked_kspace, eps = 1e-11)
            # masked_kspace1 = masked_kspace1.clamp(-6,6)
        masked_kspace1 = masked_kspace.permute(2,0,1)

    
        images = fastmri.ifft2c(masked_kspace)
        #REAL IMAGE
        image_real = fastmri.complex_abs(images)
        image_real, real_mean, real_std = T.normalize_instance(image_real, eps = 1e-11)
        image_real = image_real.clamp(-6,6)
        
        #COMPLEX IMAGE
        image, mean, std = T.normalize_instance(images, eps = 1e-11)
        image = image.clamp(-6,6) ## force between -6, 6 don't know why 
        image = image.permute(2,0,1)
        
        
        
        target_torch = T.to_tensor(target) 
        # target_torch1 = T.normalize(target_torch, mean1, std1, eps = 1e-11)
        # target_torch1= target_torch1.clamp(-6,6)
        target_torch1 = target_torch.permute(2,0,1)
        
        target_torch_images= fastmri.ifft2c(target_torch)
        target_torch_image = T.normalize(target_torch_images, mean, std, eps = 1e-11) ## change this information.
        target_torch_image = target_torch_image.clamp(-6, 6)
        target_torch_image = target_torch_image.permute(2,0,1)
        
        target_torch_real = fastmri.complex_abs(target_torch_images)
        target_torch_real = T.normalize(target_torch_real, real_mean, real_std, eps = 1e-11)
        target_torch_real = target_torch_real.clamp(-6,6)
            # else: 
            #     target_torch = torch.Tensor([0])
        
        if self.platform == 'lightning': 
            if self.objective == 'kspace': 
                return T.UnetSample(
                image = masked_kspace1, 
                target = target_torch1, 
                mean = mean,
                std = std, 
                fname = fname,
                slice_num = slice_num, 
                max_value = max_value
            )
            elif self.objective == 'complex': 
                return T.UnetSample(
                image = image, 
                target = target_torch_image, 
                mean = mean,
                std = std, 
                fname = fname,
                slice_num = slice_num, 
                max_value = max_value
            )
            elif self.objective == 'real': 
                return T.UnetSample( 
                    image = image_real, 
                    target = target_torch_real,
                    mean = real_mean, 
                    std = real_std,
                    fname = fname,
                    slice_num = slice_num, 
                    max_value = max_value)
        if self.platform == 'pytorch': 
            if self.objective == 'kspace': 
                return [masked_kspace1, target_torch1]
            elif self.objective == 'complex': 
                return [image, target_torch_image]
            elif self.objective == 'real': 
                return [image_real, target_torch_real]

class FastMRIRawDataSample(NamedTuple):
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]


class CustomUnetDataTransform(UnetDataTransform) :
    """
    UnetDataTransform for pytorch platform 
    """
    def __call__(self, kspace ,mask, target, attrs, fname, slice_num): 
        kspace_torch = T.to_tensor(kspace)
        if self.mask_func: 
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace = T.apply_mask(kspace_torch, self.mask_func, seed = seed)[0]

        else: 
            masked_kspace = kspace_torch
        image = fastmri.ifft2c(masked_kspace)
        image = fastmri.complex_abs(image)
        image, mean, std = T.normalize_instance(image, eps = 1e-11)
        image = image.clamp(-6,6) ## force between -6, 6 don't know why 
        image = image.unsqueeze(0)
        if target is not None: 
            target_torch = T.to_tensor(target)
            ## this is the original cropped kspace
            target_torch = T.normalize(target_torch, mean, std, eps = 1e-11) ## change this information.
            target_torch = target_torch.clamp(-6, 6)
            target_torch = target_torch.unsqueeze(0)
        else: 
            target_torch = torch.Tensor([0])
        return [image, target_torch]
    
class CustomSliceDataset(SliceDataset): 
    """
    Improves upon SliceDataset by supporting the "Custom" transformers
    """
    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        challenge: str,
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None,
    ): 
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.dataset_cache_file = Path(dataset_cache_file)

        self.transform = transform
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.raw_samples = []
        if raw_sample_filter is None:
            self.raw_sample_filter = lambda raw_sample: True
        else:
            self.raw_sample_filter = raw_sample_filter

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())
            for fname in sorted(files):
                metadata, num_slices = self._retrieve_metadata(fname)

                new_raw_samples = []
                for slice_ind in range(num_slices):
                    raw_sample = FastMRIRawDataSample(fname, slice_ind, metadata)
                    if self.raw_sample_filter(raw_sample):
                        new_raw_samples.append(raw_sample)

                self.raw_samples += new_raw_samples

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.raw_samples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.raw_samples = dataset_cache[root]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.raw_samples)
            num_raw_samples = round(len(self.raw_samples) * sample_rate)
            self.raw_samples = self.raw_samples[:num_raw_samples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(list(set([f[0].stem for f in self.raw_samples])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.raw_samples = [
                raw_sample
                for raw_sample in self.raw_samples
                if raw_sample[0].stem in sampled_vols
            ]

        if num_cols:
            self.raw_samples = [
                ex
                for ex in self.raw_samples
                if ex[2]["encoding_size"][1] in num_cols  # type: ignore
            ]

    
    def __len__(self):
        return len(self.raw_samples)
        
    def __getitem__(self, i: int): 
        fname, dataslice, metadata = self.raw_samples[i]

        with h5py.File(fname, "r") as hf:
            
            cropped_kspace = hf["cropped_kspace"][dataslice]
            kspace = hf["kspace"][dataslice]
            mask = np.asarray(hf["mask"]) if "mask" in hf else None
            target1 = hf["reconstruction_esc"][dataslice] 
            attrs = dict(hf.attrs)
            attrs.update(metadata)

        if self.transform is None: 
            sample = (cropped_kspace, mask, target1, attrs, fname.name, dataslice)
       # if isinstance(self.transform, KspaceDataTransform1): 
           # sample = self.transform(kspace, mask, kspace, attrs, fname.name, dataslice)
        elif isinstance(self.transform, KspaceDataTransform): ## change this 
            sample = self.transform(cropped_kspace, mask, cropped_kspace, attrs, fname.name, dataslice)
        elif isinstance(self.transform, CustomUnetDataTransform): 
            sample = self.transform(cropped_kspace, mask, target1, attrs, fname.name, dataslice)
        elif isinstance(self.transform, UnetDataTransform): 
            sample = self.transform(kspace, mask, target1, attrs, fname.name, dataslice)
        
        return sample

from torch.nn import functional as F
import pathlib

from collections import defaultdict
from fastmri.models import Unet
import pytorch_lightning as pl
import torch
from torchmetrics.metric import Metric

import fastmri
from fastmri import evaluate


class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch

    def compute(self):
        return self.quantity


class MriModule(pl.LightningModule):
    """
    MriModule now logs the input image too in tensorboard 
    validation_step_end was the modified method

    Abstract super class for deep learning reconstruction models.
    This is a subclass of the LightningModule class from pytorch_lightning,
    with some additional functionality specific to fastMRI:
        - Evaluating reconstructions
        - Visualization
    To implement a new reconstruction model, inherit from this class and
    implement the following methods:
        - training_step, validation_step, test_step:
            Define what happens in one step of training, validation, and
            testing, respectively
        - configure_optimizers:
            Create and return the optimizers
    Other methods from LightningModule can be overridden as needed.
    """

    def __init__(self, num_log_images: int = 16):
        """
        Args:
            num_log_images: Number of images to log. Defaults to 16.
        """
        super().__init__()

        self.num_log_images = num_log_images
        self.val_log_indices = None

        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()

    def validation_step_end(self, val_logs):
        # check inputs
        for k in (
            "batch_idx",
            "fname",
            "slice_num",
            "max_value",
            "output",
            "target",
            "val_loss",
            "input_image"
        ):
            if k not in val_logs.keys():
                raise RuntimeError(
                    f"Expected key {k} in dict returned by validation_step."
                )
        if val_logs["output"].ndim == 2:
            val_logs["output"] = val_logs["output"].unsqueeze(0)
        elif val_logs["output"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
        if val_logs["target"].ndim == 2:
            val_logs["target"] = val_logs["target"].unsqueeze(0)
        elif val_logs["target"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")

        # pick a set of images to log if we don't have one already
        if self.val_log_indices is None:
            self.val_log_indices = list(
                np.random.permutation(len(self.trainer.val_dataloaders[0]))[
                    : self.num_log_images
                ]
            )

        # log images to tensorboard
        if isinstance(val_logs["batch_idx"], int):
            batch_indices = [val_logs["batch_idx"]]
        else:
            batch_indices = val_logs["batch_idx"]
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                key = f"val_images_idx_{batch_idx}"
                target = val_logs["target"][i].unsqueeze(0)
                output = val_logs["output"][i].unsqueeze(0)
                input_image = val_logs["input_image"][i].unsqueeze(0)
                error = torch.abs(target - output)
                output = output / output.max()
                target = target / target.max()
                input_image = input_image/ input_image.max()
                error = error / error.max()
                self.log_image(f"{key}/target", target)
                self.log_image(f"{key}/reconstruction", output)
                self.log_image(f"{key}/error", error)
                self.log_image(f"{key}/input_image", input_image)

        # compute evaluation metrics
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()
        for i, fname in enumerate(val_logs["fname"]):
            slice_num = int(val_logs["slice_num"][i].cpu())
            maxval = val_logs["max_value"][i].cpu().numpy()
            output = val_logs["output"][i].cpu().numpy()
            target = val_logs["target"][i].cpu().numpy()

            mse_vals[fname][slice_num] = torch.tensor(
                evaluate.mse(target, output)
            ).view(1)
            target_norms[fname][slice_num] = torch.tensor(
                evaluate.mse(target, np.zeros_like(target))
            ).view(1)
            ssim_vals[fname][slice_num] = torch.tensor(
                evaluate.ssim(target[None, ...], output[None, ...], maxval=maxval)
            ).view(1)
            max_vals[fname] = maxval

        return {
            "val_loss": val_logs["val_loss"],
            "mse_vals": dict(mse_vals),
            "target_norms": dict(target_norms),
            "ssim_vals": dict(ssim_vals),
            "max_vals": max_vals,
        }

    def log_image(self, name, image):
        self.logger.experiment.add_image(name, image, global_step=self.global_step)

    def validation_epoch_end(self, val_logs):
        # aggregate losses
        losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        # use dict updates to handle duplicate slices
        for val_log in val_logs:
            losses.append(val_log["val_loss"].view(-1))

            for k in val_log["mse_vals"].keys():
                mse_vals[k].update(val_log["mse_vals"][k])
            for k in val_log["target_norms"].keys():
                target_norms[k].update(val_log["target_norms"][k])
            for k in val_log["ssim_vals"].keys():
                ssim_vals[k].update(val_log["ssim_vals"][k])
            for k in val_log["max_vals"]:
                max_vals[k] = val_log["max_vals"][k]

        # check to make sure we have all files in all metrics
        assert (
            mse_vals.keys()
            == target_norms.keys()
            == ssim_vals.keys()
            == max_vals.keys()
        )

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals[fname].items()])
            )
            target_norm = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
            )
            metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
            metrics["psnr"] = (
                metrics["psnr"]
                + 20
                * torch.log10(
                    torch.tensor(
                        max_vals[fname], dtype=mse_val.dtype, device=mse_val.device
                    )
                )
                - 10 * torch.log10(mse_val)
            )
            metrics["ssim"] = metrics["ssim"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))
        val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        tot_slice_examples = self.TotSliceExamples(
            torch.tensor(len(losses), dtype=torch.float)
        )

        self.log("validation_loss", val_loss / tot_slice_examples, prog_bar=True, sync_dist=True)
        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}", value / tot_examples, sync_dist=True)

    def test_epoch_end(self, test_logs):
        outputs = defaultdict(dict)

        # use dicts for aggregation to handle duplicate slices in ddp mode
        for log in test_logs:
            for i, (fname, slice_num) in enumerate(zip(log["fname"], log["slice"])):
                outputs[fname][int(slice_num.cpu())] = log["output"][i]

        # stack all the slices for each file
        for fname in outputs:
            outputs[fname] = np.stack(
                [out for _, out in sorted(outputs[fname].items())]
            )

        # pull the default_root_dir if we have a trainer, otherwise save to cwd
        if hasattr(self, "trainer"):
            save_path = pathlib.Path(self.trainer.default_root_dir) / "reconstructions"
        else:
            save_path = pathlib.Path.cwd() / "reconstructions"
        self.print(f"Saving reconstructions to {save_path}")

        fastmri.save_reconstructions(outputs, save_path)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # logging params
        parser.add_argument(
            "--num_log_images",
            default=16,
            type=int,
            help="Number of images to log to Tensorboard",
        )

        return parser
    
    
    
    
class UnetModule(MriModule):
    """
    UnetModule now supports training with 2 channel input images in addition
    to standard 1 input channel real image training.
    Compared to original UnetModule the following methods were changed:
        forward 
        validation_step 

    Unet training module.
    This can be used to train baseline U-Nets from the paper:
    J. Zbontar et al. fastMRI: An Open Dataset and Benchmarks for Accelerated
    MRI. arXiv:1811.08839. 2018.
    """

    def __init__(
        self,
        approach = "real",
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        **kwargs,
    ):
        """
        Args:
            approach: training either on the "real" image or on "complex" image 
                or the complex "kspace" (metrics are calculated on the 
                reconstructed real image). Defaults to real. 
            in_chans (int, optional): Number of channels in the input to the
                U-Net model. Defaults to 1.
            out_chans (int, optional): Number of channels in the output to the
                U-Net model. Defaults to 1.
            chans (int, optional): Number of output channels of the first
                convolution layer. Defaults to 32.
            num_pool_layers (int, optional): Number of down-sampling and
                up-sampling layers. Defaults to 4.
            drop_prob (float, optional): Dropout probability. Defaults to 0.0.
            lr (float, optional): Learning rate. Defaults to 0.001.
            lr_step_size (int, optional): Learning rate step size. Defaults to
                40.
            lr_gamma (float, optional): Learning rate gamma decay. Defaults to
                0.1.
            weight_decay (float, optional): Parameter for penalizing weights
                norm. Defaults to 0.0.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        if approach not in ['real', 'complex', 'kspace']: 
            ValueError('approach should be either "real" or "complex" or "kspace"')
                            
        self.approach = approach
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.unet = Unet(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            chans=self.chans,
            num_pool_layers=self.num_pool_layers,
            drop_prob=self.drop_prob,
        )

    def forward(self, image):
        if image.dim() == 3: 
            return self.unet(image.unsqueeze(1)).squeeze(1)
        if image.dim() == 4: 
            return self.unet(image)

    def training_step(self, batch, batch_idx):
        output = self(batch.image)
        loss = F.l1_loss(output, batch.target)
        
        self.log("loss", loss.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        # print(f" input type = {batch.image.dtype}, input = {batch.image}")
        output = self(batch.image)
        

        image = batch.image## input
        if output.dim() == 3: 
            mean = batch.mean.unsqueeze(1).unsqueeze(2) 
            std = batch.std.unsqueeze(1).unsqueeze(2)
        if output.dim() == 4: 
            mean = batch.mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            std = batch.std.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # print(mean.shape, std.shape, mean)
        if self.approach == 'real': 

            image = image * std + mean
            output1 = output*std + mean
            target = batch.target*std + mean
        if self.approach == 'complex': 
            output1 = output.permute(0,2,3,1)
            # print(f" output + {output.shape}")
            output1 = output1*std + mean
            
            output1 = fastmri.complex_abs(output1)
            image = image.permute(0,2,3,1)
            image = image*std + mean 
            image = fastmri.complex_abs(image)
            target = batch.target.permute(0,2,3,1)
            target = target*std + mean
            target= fastmri.complex_abs(target)

        if self.approach == 'kspace':
            output1 = output.permute(0,2,3,1)
            # output1 = output1 * std + mean
            output1 = fastmri.complex_abs(fastmri.ifft2c(output1))
            output1 = T.center_crop(output1, (320,320))
            image = image.permute(0,2,3,1)
            # image = image*std + mean 
            image = fastmri.complex_abs(fastmri.ifft2c(image))
            image = T.center_crop(image,(320,320))
            target = batch.target.permute(0,2,3,1)
            # target = target * std + mean
            target = fastmri.complex_abs(fastmri.ifft2c(target))
            target = T.center_crop(target, (320,320))
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output1,
            "target": target,
            "val_loss": F.l1_loss(output, batch.target),
            "input_image": image
        }

    def test_step(self, batch, batch_idx):
        output = self.forward(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": (output * std + mean).cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # network params
        parser.add_argument("--approach", default = 'real')
        parser.add_argument(
            "--in_chans", default=1, type=int, help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=1, type=int, help="Number of U-Net output chanenls"
        )
        parser.add_argument(
            "--chans", default=1, type=int, help="Number of top-level U-Net filters."
        )
        parser.add_argument(
            "--num_pool_layers",
            default=4,
            type=int,
            help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--drop_prob", default=0.0, type=float, help="U-Net dropout probability"
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.001, type=float, help="RMSProp learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma", default=0.1, type=float, help="Amount to decrease step size"
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser