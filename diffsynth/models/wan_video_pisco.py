import torch
import torch.nn as nn
from .wan_video_dit import DiTBlock

class PISCOWanAttentionBlock(DiTBlock):
    def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps=1e-6, block_id=0):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = torch.nn.Linear(self.dim, self.dim)
        self.after_proj = torch.nn.Linear(self.dim, self.dim)

    def forward(self, c, x, context, t_mod, freqs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, context, t_mod, freqs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c

class PISCOWanModel(torch.nn.Module):
    def __init__(
        self,
        pisco_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
        pisco_in_dim=132,
        patch_size=(1, 2, 2),
        has_image_input=False,
        dim=1536,
        num_heads=12,
        ffn_dim=8960,
        eps=1e-6,
    ):
        super().__init__()
        self.pisco_layers = pisco_layers
        self.pisco_in_dim = pisco_in_dim
        self.pisco_layers_mapping = {i: n for n, i in enumerate(self.pisco_layers)}

        # Renamed: vace_blocks -> pisco_blocks
        self.pisco_blocks = torch.nn.ModuleList([
            PISCOWanAttentionBlock(has_image_input, dim, num_heads, ffn_dim, eps, block_id=i)
            for i in self.pisco_layers
        ])

        # Renamed: vace_patch_embedding -> pisco_patch_embedding
        self.pisco_patch_embedding = torch.nn.Conv3d(pisco_in_dim, dim, kernel_size=patch_size, stride=patch_size)

    def forward(
        self, x, pisco_context, context, t_mod, freqs,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        # x: torch.Size([1, 20280, 1536])
        # pisco_context: torch.Size([1, pisco_in_dim, 13, 60, 104])
        
        # Use pisco_patch_embedding
        c = [self.pisco_patch_embedding(u.unsqueeze(0)) for u in pisco_context] 
        c = [u.flatten(2).transpose(1, 2) for u in c] 
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, x.shape[1] - u.size(1), u.size(2))],
                      dim=1) for u in c
        ]) 
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        # Use pisco_blocks
        for block in self.pisco_blocks:
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    c = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        c, x, context, t_mod, freqs,
                        use_reentrant=True,
                    )
            elif use_gradient_checkpointing:
                c = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    c, x, context, t_mod, freqs,
                    use_reentrant=True,
                )
            else:
                c = block(c, x, context, t_mod, freqs)
        hints = torch.unbind(c)[:-1]
        return hints