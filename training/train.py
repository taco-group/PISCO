import torch, os, argparse, accelerate, warnings
import wandb
import re
import uuid
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from diffsynth.core import UnifiedDataset, find_recent_checkpoint
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath, LoadVideoWithMask
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.core import load_state_dict
from utils.data_operators import LoadVideoKeepFirstFrame, LoadVideoAllFrames
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# import deepspeed
# torch.utils.checkpoint.checkpoint = deepspeed.checkpointing.checkpoint


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        auto_load_checkpoints=False,
        model_checkpoint_path=None,
        output_path=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        loading_model=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else (
            ModelConfig(model_id=tokenizer_path.split(":")[0], origin_file_pattern=tokenizer_path.split(":")[1]) if ":" in tokenizer_path else ModelConfig(tokenizer_path)
        )
        audio_processor_config = ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/") if audio_processor_path is None else ModelConfig(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)

        # <--- Mod 2: Initialize step ---
        self.current_step = 0

        def load_modules(modules):
            if ',' in modules:
                for module in modules.split(','):
                    self.pipe.get_submodule(module).load_state_dict(state_dict, strict=False)
            else:
                self.pipe.get_submodule(modules).load_state_dict(state_dict, strict=False)
        
        if model_checkpoint_path is not None:
            state_dict = load_state_dict(model_checkpoint_path)
            if loading_model is not None:
                load_modules(loading_model)
            else:
                load_modules(trainable_models)
            # <--- Mod 3: Parse step ---
            self.current_step = self.parse_step_from_path(model_checkpoint_path)
            
        elif auto_load_checkpoints:
            ckp = find_recent_checkpoint(output_path)
            if ckp is not None:
                print(f"Auto loading checkpoint: {ckp}")
                state_dict = load_state_dict(ckp)
                if loading_model is not None:
                    load_modules(loading_model)
                else:
                    load_modules(trainable_models)
                print(f"Auto loaded checkpoint: {ckp}")
                # <--- Mod 4: Parse step ---
                self.current_step = self.parse_step_from_path(ckp)
            else:
                print("Checkpoints not found. Ready to be trained from stratch.")

        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
    
    # <--- Mod 5: New parser function ---
    def parse_step_from_path(self, path):
        try:
            filename = os.path.basename(path)
            numbers = re.findall(r'(\d+)', filename)
            if numbers: return int(numbers[-1])
        except: pass
        return 0

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,

            "pisco_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        with torch.no_grad():
            for unit in self.pipe.units:
                inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--pisco_first_frame_only", default=False, action="store_true",
                        help="Use first-frame-only temporal masking for pisco_reference_video/depth. "
                             "For instance insertion training where reference is only at frame 0.")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
        log_with="wandb"
    )


    slurm_id = os.environ.get("SLURM_JOB_ID") 
    wandb_id = slurm_id if slurm_id else uuid.uuid4().hex[:8]  # wandb.util.generate_id() is gone in recent wandb
    wandb_init_kwargs = {
        "wandb": {
            "id": wandb_id,
            "resume": "allow",
            "name": f"run_{wandb_id}",  # Optional: set run name to id for easy lookup
            "notes": "Video Instance Editing / Propagation Training"
        }
    }
    accelerator.init_trackers(
        project_name="PISCO",         # Corresponds to your original project="vise"
        config=vars(args),           # Record hyperparameters
        init_kwargs=wandb_init_kwargs # Key: inject your slurm logic
    )

    base_op = ToAbsolutePath(args.dataset_base_path)
    masked_loader_op = base_op >> LoadVideoWithMask(
        num_frames=args.num_frames, 
        # keep_num=args.num_frames * args.pisco_keep_ratio, 
        height=args.height, width=args.width, max_pixels=args.max_pixels, 
        height_division_factor=16, width_division_factor=16
    )

    if args.pisco_first_frame_only:
        first_frame_op = base_op >> LoadVideoKeepFirstFrame(
            num_frames=args.num_frames,
            height=args.height, width=args.width, max_pixels=args.max_pixels,
            height_division_factor=16, width_division_factor=16
        )
        all_frames_op = base_op >> LoadVideoAllFrames(
            num_frames=args.num_frames,
            height=args.height, width=args.width, max_pixels=args.max_pixels,
            height_division_factor=16, width_division_factor=16
        )
        pisco_ref_op = first_frame_op
        pisco_mask_op = all_frames_op
        pisco_depth_op = all_frames_op
    else:
        pisco_ref_op = masked_loader_op
        pisco_mask_op = masked_loader_op
        pisco_depth_op = masked_loader_op

    # Load dataset
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map = {
            "animate_face_video":     base_op >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio":            base_op >> LoadAudio(sr=16000),

            "pisco_reference_depth":   pisco_ref_op,
            "pisco_reference_video":   pisco_ref_op,
            "pisco_video_mask":        pisco_mask_op,
            "pisco_depth":             pisco_depth_op,
            "num_inference_steps":    lambda x: x,
        }
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        auto_load_checkpoints=args.auto_load_checkpoints,
        model_checkpoint_path=args.model_checkpoint_path,
        output_path=args.output_path,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        loading_model=args.loading_model,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    
    if model.current_step > 0:
        model_logger.num_steps = model.current_step
        args.initial_global_step = model.current_step # Must be preserved to prevent LR confusion
        print(f"Resuming from step {model.current_step}")

    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "dmp_sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)