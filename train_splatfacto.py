from collections import defaultdict, OrderedDict
import functools
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.callbacks import TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.process_data.images_to_nerfstudio_dataset import ImagesToNerfstudioDataset
import numpy as np
from pathlib import Path
import time
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from typing import DefaultDict, Dict, Literal, Tuple


checkpoint_dir: Path = Path("path/to/checkpoint/dir/")
# Directory to save checkpoints
colmap_cmd: str = "colmap"
# Command to run COLMAP, e.g. "colmap" or "path/to/colmap/colmap"
sfm_matching_method: Literal["exhaustive", "sequential", "vocab_tree"] = "sequential"
# Use sequential for continuous video frames, default to vocab_tree otherwise
sfm_output_dir: Path = Path("path/to/output/dir/") # Spaces not allowed
# Output directory for the COLMAP reconstruction
data_dir: Path = Path("path/to/data/Images/") # Spaces not allowed
# Directory containing the images to be processed
num_iterations: int = 7_000
# Number of iterations to train the model for
output_splat_filename: Path = Path("path/to/output/splatfacto.splat")
# Output filename for the .splat file
sfm_tool: Literal['any', 'colmap', 'hloc'] = 'colmap'
# Which tool to use for the reconstruction, default to colmap


def sh2rgb(sh: list[float]) -> np.ndarray:
    """
    Converts from 0th order spherical harmonics to rgb [0, 255]
    """
    C0 = 0.28209479177387814
    rgb = [sh[i] * C0 + 0.5 for i in range(len(sh))]
    return np.clip(rgb, 0, 1) * 255


splatfacto_config = TrainerConfig(
    method_name="splatfacto",
    max_num_iterations=num_iterations,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=FullImageDatamanagerConfig(
            dataparser=NerfstudioDataParserConfig(data=sfm_output_dir, load_3D_points=True),
            cache_images_type="uint8",
        ),
        model=SplatfactoModelConfig(),
    ),
    optimizers={
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=30000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {
            "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
            "scheduler": None
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
    }
)


class Trainer:
    """
    This is modified from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/engine/trainer.py
    
    Trainer class

    Args:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.

    Attributes:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.
        device: The device to run the training on.
        pipeline: The pipeline object.
        optimizers: The optimizers object.
        training_state: Current model training state.
    """
    
    pipeline: VanillaPipeline
    optimizers: Optimizers
    output_splat_filename: Path
    
    def __init__(self, config: TrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.device: str = config.machine.device_type
        if self.device == "cuda":
            self.device += f":{local_rank}"
        self.mixed_precision: bool = self.config.mixed_precision
        self.use_grad_scaler: bool = self.mixed_precision or self.config.use_grad_scaler
        self.gradient_accumulation_steps: DefaultDict = defaultdict(lambda: 1)
        self.gradient_accumulation_steps.update(self.config.gradient_accumulation_steps)
        
        if self.device == "cpu":
            self.mixed_precision = False
            print("Mixed precision is disabled for CPU training.")
        self._start_step: int = 0
        # optimizers
        self.grad_scaler = GradScaler(enabled=self.use_grad_scaler)
        
        self.step = 0
        
        self.base_dir: Path = config.get_base_dir()
        self.checkpoint_dir: Path = config.get_checkpoint_dir()
        
    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datasets into memory
                'inference': does not load any dataset into memory
        """
        self.pipeline = self.config.pipeline.setup(
            device=self.device,
            test_mode=test_mode,
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
        )
        self.optimizers = self.setup_optimizers()
        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers, grad_scaler=self.grad_scaler, pipeline=self.pipeline, trainer=self
            )
        )
        
    def setup_optimizers(self) -> Optimizers:
        """Helper to set up the optimizers

        Returns:
            The optimizers object given the trainer config.
        """
        optimizer_config = self.config.optimizers.copy()
        param_groups = self.pipeline.get_param_groups()
        return Optimizers(optimizer_config, param_groups)
        

    def train_iteration(self, step: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """
        needs_zero = [
            group for group in self.optimizers.parameters.keys() if step % self.gradient_accumulation_steps[group] == 0
        ]
        self.optimizers.zero_grad_some(needs_zero)
        cpu_or_cuda_str: str = self.device.split(":")[0]
        cpu_or_cuda_str = "cpu" if cpu_or_cuda_str == "mps" else cpu_or_cuda_str

        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
            loss = functools.reduce(torch.add, loss_dict.values())
        self.grad_scaler.scale(loss).backward()  # type: ignore
        needs_step = [
            group
            for group in self.optimizers.parameters.keys()
            if step % self.gradient_accumulation_steps[group] == self.gradient_accumulation_steps[group] - 1
        ]
        self.optimizers.optimizer_scaler_step_some(self.grad_scaler, needs_step)

        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad  # type: ignore
                    total_grad += grad

            metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)  # type: ignore

        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            self.optimizers.scheduler_step_all(step)
        
        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict  # type: ignore

    def train(self) -> None:
        """
        Trains a splatfacto model on the provided data
        """
        for step in range(self.step + 1, self.config.max_num_iterations + 1 - self.step):
            self.step = step
            if step % 100 == 0:
                print(f'Step: {step}')
            self.pipeline.train()
            
            # training callbacks before the training iteration
            for callback in self.callbacks:
                callback.run_callback_at_location(
                    step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                )
            
            loss, loss_dict, metrics_dict = self.train_iteration(step)
            
            # training callbacks after the training iteration
            for callback in self.callbacks:
                callback.run_callback_at_location(
                    step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                )
            
    def export_splat(self, output_splat_filename: Path) -> None:
        """
        Saves the (ideally) trained splat to the provided filename as a .splat file. 
        Adapted from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/scripts/exporter.py#L478
        and my ply_to_splat.py file
        """
        
        model: SplatfactoModel = self.pipeline.model
        
        count = 0
        map_to_tensors = OrderedDict()
        
        with torch.no_grad():
            positions = model.means.cpu().numpy()
            count = positions.shape[0]
            n = count
            map_to_tensors["x"] = positions[:, 0]
            map_to_tensors["y"] = positions[:, 1]
            map_to_tensors["z"] = positions[:, 2]
            
            shs_0 = model.shs_0.contiguous().cpu().numpy()
            for i in range(shs_0.shape[1]):
                map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

            map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()
            
            scales = model.scales.data.cpu().numpy()
            for i in range(3):
                map_to_tensors[f"scale_{i}"] = scales[:, i, None]

            quats = model.quats.data.cpu().numpy()
            for i in range(4):
                map_to_tensors[f"rot_{i}"] = quats[:, i, None]
                
        # post optimization, it is possible have NaN/Inf values in some attributes
        # to ensure the exported ply file has finite values, we enforce finite filters.
        select = np.ones(n, dtype=bool)
        for k, t in map_to_tensors.items():
            n_before = np.sum(select)
            select = np.logical_and(select, np.isfinite(t).all(axis=-1))
            n_after = np.sum(select)
            if n_after < n_before:
                print(f"{n_before - n_after} NaN/Inf elements in {k}")

        if np.sum(select) < n:
            print(f"values have NaN/Inf in map_to_tensors, only export {np.sum(select)}/{n}")
            for k, t in map_to_tensors.items():
                map_to_tensors[k] = map_to_tensors[k][select]
            count = np.sum(select)
        
        with open(output_splat_filename, "wb") as splat_file:
            for i in range(count):
                # Position
                splat_file.write(map_to_tensors['x'][i].tobytes())
                splat_file.write(map_to_tensors['y'][i].tobytes())
                splat_file.write(map_to_tensors['z'][i].tobytes())
                
                # Scale
                for j in range(3):
                    splat_file.write(np.exp(map_to_tensors[f'scale_{j}'][i]).tobytes())
                 
                # Color
                sh = [map_to_tensors[f"f_dc_{j}"][i] for j in range(3)]
                rgb = sh2rgb(sh)
                for color in rgb:
                    splat_file.write(color.astype(np.uint8).tobytes())
                
                # Opacity
                opac = 1.0 + np.exp(-map_to_tensors['opacity'][i])
                opacity = np.clip((1.0/opac) * 255, 0, 255)
                splat_file.write(opacity.astype(np.uint8).tobytes())
                
                # Quaternion rotation
                rot = np.array([map_to_tensors[f"rot_{j}"][i] for j in range(4)])
                rot = np.clip(rot * 128 + 128, 0, 255)
                for j in range(4):
                    splat_file.write(rot[j].astype(np.uint8).tobytes())
                    
    def save_checkpoint(self) -> None: # This is for debugging purposes only
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """
        
        self.config.save_config()
        
        self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(
            self.base_dir / "dataparser_transforms.json"
        )
        
        # possibly make the checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path: Path = self.checkpoint_dir / f"step-{self.step:09d}.ckpt"
        torch.save(
            {
                "step": self.step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else self.pipeline.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
            
            
if __name__ == "__main__":
    
    start_time = time.time()
    
    colmap_start_time = time.time()
    
    print('\nRunning COLMAP...\n')
    # Processing the images with colmap
    images_to_nerfstudio_dataset = ImagesToNerfstudioDataset(
        colmap_cmd=colmap_cmd,
        data=data_dir,
        matching_method=sfm_matching_method,
        output_dir=sfm_output_dir,
        sfm_tool=sfm_tool
    )
    images_to_nerfstudio_dataset.main()
    
    colmap_end_time = time.time()
    print(f'\nCOLMAP took {colmap_end_time - colmap_start_time} seconds')
    
    training_start_time = time.time()
    
    print('\nOptimizing Gaussian splats...\n')
    config = splatfacto_config
    trainer = Trainer(config)
    trainer.setup()
    trainer.train()
    
    print(f'\nSuccessfully optimized the Gaussian splats!')
    
    training_end_time = time.time()
    print(f'\nTraining took {training_end_time - training_start_time} seconds')
    
    # Saving checkpoint for debugging
    # trainer.save_checkpoint()
    
    export_start_time = time.time()
    
    print('\nExporting splats...')
    trainer.export_splat(output_splat_filename)
    print(f'\nSuccessfully saved the .splat file!')
    
    export_end_time = time.time()
    print(f'\nExporting took {export_end_time - export_start_time} seconds')
    
    end_time = time.time()
    print(f"\nThe overall time was {end_time - start_time} seconds")