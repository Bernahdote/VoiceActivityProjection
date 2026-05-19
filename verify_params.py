import torch
from pathlib import Path

checkpoint_path = Path("/mnt/sdb/willem/VoiceActivityProjection/runs_new/VAP_debug/cp009hag/checkpoints/epoch=6-step=22505.ckpt")
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

if isinstance(ckpt, dict) and "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    state_dict = ckpt

# Count by component
encoder_params = 0
transformer_params = 0
head_params = 0
other_params = 0

for key, val in state_dict.items():
    if not isinstance(val, torch.Tensor):
        continue
    
    params = val.numel()
    
    if "encoder" in key:
        encoder_params += params
    elif "transformer" in key:
        transformer_params += params
    elif "va_classifier" in key or "vap_head" in key:
        head_params += params
    else:
        other_params += params

total = encoder_params + transformer_params + head_params + other_params

print(f"Encoder params:      {encoder_params:>12,}")
print(f"Transformer params:  {transformer_params:>12,}")
print(f"Head params:         {head_params:>12,}")
print(f"Other params:        {other_params:>12,}")
print(f"{'─' * 40}")
print(f"Total:               {total:>12,}")
print(f"\nExpected breakdown:")
print(f"  Trainable (from model instantiation): 3,939,585")
print(f"  Difference: {total - 3939585:,}")
