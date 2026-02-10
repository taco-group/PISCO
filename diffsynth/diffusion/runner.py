import os
import torch
from tqdm import tqdm
from accelerate import Accelerator

from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


# =========================
# ZeRO-3: runtime ignore VAE
# =========================

class _AttrDict(dict):
    """dict subclass that supports attribute assignment (for DeepSpeed _in_forward)."""
    pass


class VAEProxyModule(torch.nn.Module):
    """
    - Registered in pipe.vae (so pipe.vae.model.z_dim etc. works)
    - Real VAE exists in self.__dict__['_real_vae'] (not registered as submodule/parameter) => ZeRO-3 won't shard real VAE
    - Proxy has a dummy param to ensure DS hook/flow works
    - Critical: Do not instantiate this module again after prepare() (otherwise ds_* params won't be initialized)
    """
    def __init__(self, real_vae: torch.nn.Module, device: torch.device, use_autocast: bool = True):
        super().__init__()

        # dummy param: must be requires_grad=True, ensures some DS versions handle this module
        self.register_parameter("_ds_dummy", torch.nn.Parameter(torch.zeros(1), requires_grad=True))

        # Allow DS to write module._parameters._in_forward
        if type(self._parameters) is dict:
            self.__dict__["_parameters"] = _AttrDict(self._parameters)

        # Do not register real_vae as submodule
        self.__dict__["_real_vae"] = real_vae
        self.__dict__["_device"] = device
        self.__dict__["_use_autocast"] = use_autocast

    def set_device(self, device: torch.device):
        """Update device for each rank after prepare: Do not create new params/modules!"""
        self.__dict__["_device"] = device

    def set_use_autocast(self, use_autocast: bool):
        self.__dict__["_use_autocast"] = use_autocast

    def _maybe_to_device(self):
        vae = self.__dict__["_real_vae"]
        device = self.__dict__["_device"]
        params = list(vae.parameters())
        if len(params) == 0:
            vae.to(device)
            return
        if params[0].device != device:
            vae.to(device)

    @torch.no_grad()
    def encode(self, *args, **kwargs):
        self._maybe_to_device()
        vae = self.__dict__["_real_vae"]
        device = self.__dict__["_device"]
        if self.__dict__["_use_autocast"] and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return vae.encode(*args, **kwargs)
        return vae.encode(*args, **kwargs)

    @torch.no_grad()
    def decode(self, *args, **kwargs):
        self._maybe_to_device()
        vae = self.__dict__["_real_vae"]
        device = self.__dict__["_device"]
        if self.__dict__["_use_autocast"] and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return vae.decode(*args, **kwargs)
        return vae.decode(*args, **kwargs)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        self._maybe_to_device()
        vae = self.__dict__["_real_vae"]
        device = self.__dict__["_device"]
        if self.__dict__["_use_autocast"] and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return vae(*args, **kwargs)
        return vae(*args, **kwargs)

    def __getattr__(self, name):
        if name in ("_real_vae", "_device", "_use_autocast"):
            return self.__dict__[name]
        vae = self.__dict__.get("_real_vae", None)
        if vae is None:
            raise AttributeError(name)
        return getattr(vae, name)


def detach_real_vae(pipe: torch.nn.Module, vae_name: str = "vae") -> torch.nn.Module:
    """Extract real VAE from pipe.vae and freeze it."""
    real_vae = getattr(pipe, vae_name)
    if real_vae is None:
        raise ValueError(f"pipe.{vae_name} is already None")
    real_vae.eval()
    for p in real_vae.parameters():
        p.requires_grad_(False)
    return real_vae


def install_vae_proxy_once(
    pipe: torch.nn.Module,
    real_vae: torch.nn.Module,
    init_device: torch.device,
    vae_name: str = "vae",
    use_autocast: bool = True,
    force_place: str = "cuda",
):
    """
    Call only once before prepare: replace pipe.vae with proxy (containing dummy param)
    Do NOT new this proxy again after prepare.
    """
    if force_place == "cpu":
        init_device = torch.device("cpu")
        real_vae.to(init_device)
    else:
        real_vae.to(init_device)

    proxy = VAEProxyModule(real_vae, device=init_device, use_autocast=use_autocast)
    setattr(pipe, vae_name, proxy)
    return proxy


def update_proxy_runtime_config_after_prepare(
    model,
    accelerator_device: torch.device,
    place_vae_on: str,
    use_vae_autocast: bool,
    pipe_attr: str = "pipe",
    vae_name: str = "vae",
):
    """Update proxy device/amp after prepare, do not create new params."""
    pipe = getattr(model, pipe_attr)
    proxy = getattr(pipe, vae_name)
    assert isinstance(proxy, VAEProxyModule), "pipe.vae is not VAEProxyModule; check installation"

    if place_vae_on == "cpu":
        proxy.set_device(torch.device("cpu"))
    else:
        proxy.set_device(accelerator_device)

    proxy.set_use_autocast(use_vae_autocast)


def assert_zero3_ignores_real_vae(model, pipe_attr: str = "pipe", vae_name: str = "vae"):
    """Verify real VAE params are not in model.parameters()."""
    pipe = getattr(model, pipe_attr)
    proxy = getattr(pipe, vae_name)
    assert isinstance(proxy, VAEProxyModule), "pipe.vae should be VAEProxyModule"

    real_vae = proxy.__dict__.get("_real_vae", None)
    assert real_vae is not None, "proxy missing _real_vae"

    vae_param_ids = {id(p) for p in real_vae.parameters()}
    model_param_ids = {id(p) for p in model.parameters()}
    assert len(vae_param_ids & model_param_ids) == 0, "Real VAE params still in model.parameters()!"


# =========================
# Training / Data processing
# =========================

def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args=None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs

    warmup_steps = args.warmup_steps if hasattr(args, "warmup_steps") else 0
    initial_global_step = getattr(args, "initial_global_step", 0) if args else 0

    place_vae_on = getattr(args, "vae_device", "cuda")      # "cuda" or "cpu"
    use_vae_autocast = getattr(args, "vae_autocast", True)  # True/False

    is_zero3 = getattr(accelerator.state, "deepspeed_plugin", None) is not None and getattr(accelerator.state.deepspeed_plugin, "zero_stage", None) == 3

    # ---- Before accelerator.prepare(): detach real VAE and install proxy ONCE ----
    pipe = getattr(model, "pipe")
    real_vae = detach_real_vae(pipe, vae_name="vae")

    # Install proxy only once before prepare; do NOT create a new proxy after prepare.
    init_device = torch.device("cpu") if place_vae_on == "cpu" else accelerator.device
    if is_zero3:
        install_vae_proxy_once(
            pipe=pipe,
            real_vae=real_vae,
            init_device=init_device,
            vae_name="vae",
            use_autocast=use_vae_autocast,
            force_place=place_vae_on,
        )

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    for group in optimizer.param_groups:
        if "initial_lr" not in group:
            group["initial_lr"] = group["lr"]

    if warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0),
            last_epoch=0,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, last_epoch=initial_global_step - 1)

    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers
    )

    # ---- accelerator.prepare(): DeepSpeed/ZeRO wraps the model ----
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    # ---- After prepare: only update runtime config (device/autocast), do NOT create new params ----
    if is_zero3:
        update_proxy_runtime_config_after_prepare(
            model=model,
            accelerator_device=accelerator.device,
            place_vae_on=place_vae_on,
            use_vae_autocast=use_vae_autocast,
            pipe_attr="pipe",
            vae_name="vae",
        )

    # Verify: real VAE params are NOT managed by ZeRO-3 (not in model.parameters()).
    if is_zero3:
        assert_zero3_ignores_real_vae(model, pipe_attr="pipe", vae_name="vae")

    max_grad_norm = 1.0
    global_step = initial_global_step

    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                if getattr(dataset, "load_from_cache", False):
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # Gradient clipping is only valid on sync step (true global step).
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    current_grad_norm = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
                    current_lr = optimizer.param_groups[0]["lr"]

                    accelerator.log(
                        {
                            "train/loss": float(loss.item()),
                            "train/grad_norm": float(current_grad_norm),
                            "train/lr": float(current_lr),
                            "epoch": int(epoch_id),
                        },
                        step=global_step,
                    )

                    # Increase ONLY at sync point -> consistent across ranks.
                    global_step += 1

                    # IMPORTANT:
                    # Trigger checkpoint using runner-controlled global_step (collective-safe).
                    # This prevents per-rank drift that can deadlock ZeRO-3 allgathers.
                    model_logger.on_step_end(accelerator, model, global_step, save_steps)

                optimizer.step()
                scheduler.step()

        # Epoch checkpoint (all ranks participate in get_state_dict inside logger).
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    # Final checkpoint if needed (collective-safe inside logger).
    model_logger.on_training_end(accelerator, model, global_step, save_steps)

def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args=None,
):
    if args is not None:
        num_workers = args.dataset_num_workers

    place_vae_on = getattr(args, "vae_device", "cuda")
    use_vae_autocast = getattr(args, "vae_autocast", True)

    is_zero3 = getattr(accelerator.state, "deepspeed_plugin", None) is not None and getattr(accelerator.state.deepspeed_plugin, "zero_stage", None) == 3

    pipe = getattr(model, "pipe")
    real_vae = detach_real_vae(pipe, vae_name="vae")

    init_device = torch.device("cpu") if place_vae_on == "cpu" else accelerator.device
    if is_zero3:
        install_vae_proxy_once(
            pipe=pipe,
            real_vae=real_vae,
            init_device=init_device,
            vae_name="vae",
            use_autocast=use_vae_autocast,
            force_place=place_vae_on,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers
    )

    model, dataloader = accelerator.prepare(model, dataloader)

    if is_zero3:
        update_proxy_runtime_config_after_prepare(
            model=model,
            accelerator_device=accelerator.device,
            place_vae_on=place_vae_on,
            use_vae_autocast=use_vae_autocast,
            pipe_attr="pipe",
            vae_name="vae",
        )

    if is_zero3:
        assert_zero3_ignores_real_vae(model, pipe_attr="pipe", vae_name="vae")

    for data_id, data in enumerate(tqdm(dataloader, disable=not accelerator.is_main_process)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(folder, f"{data_id}.pth")
                out = model(data)
                torch.save(out, save_path)
