import os
import torch
from accelerate import Accelerator


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, global_step: int, save_steps: int | None):
        """
        Save checkpoint by *global_step* that is controlled by the runner.
        This avoids per-rank step drift (which can deadlock ZeRO-3 collectives).
        """
        if save_steps is None:
            return
        # Important: the condition must be evaluated identically on all ranks.
        if global_step > 0 and (global_step % save_steps == 0):
            self.save_model(accelerator, model, f"step-{global_step}.safetensors")

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id: int):
        """Epoch checkpoint (all ranks participate in ZeRO-3 gather)."""
        self.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors")

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, global_step: int, save_steps: int | None):
        """
        Optionally save the last step if it is not exactly on a save boundary.
        """
        if save_steps is None:
            return
        if global_step % save_steps != 0:
            self.save_model(accelerator, model, f"step-{global_step}.safetensors")

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name: str):
        """
        ZeRO-3 safe saving pattern:
        - ALL ranks must call accelerator.get_state_dict(model) (collectives inside).
        - Only main process writes to disk.
        - Barriers before/after keep collective order aligned across ranks.
        """
        accelerator.wait_for_everyone()
        
        # Critical: every rank must participate, even if only rank0 writes the file.
        state_dict = accelerator.get_state_dict(model)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)

            # Keep only trainable parameters (your original behavior).
            state_dict = unwrapped.export_trainable_state_dict(
                state_dict,
                remove_prefix=self.remove_prefix_in_ckpt
            )

            state_dict = self.state_dict_converter(state_dict)
            state_dict = {k: v for k, v in state_dict.items() if "pipe.vae._ds_dummy" not in k}

            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)

            # Writing is main-process only.
            accelerator.save(state_dict, path, safe_serialization=True)

        accelerator.wait_for_everyone()