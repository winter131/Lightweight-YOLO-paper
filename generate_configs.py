import os

# Create configs directory
os.makedirs('configs', exist_ok=True)

# Define components
# Baseline Layer 10
layer_10 = "  - [-1, 2, C3k2, [1024, True]] # 10"

# Component A: Slim-Neck
# Original Neck
neck_standard = """head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 11
  - [[-1, 6], 1, Concat, [1]] # 12
  - [-1, 2, C3k2, [512, False]] # 13
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 14
  - [[-1, 4], 1, Concat, [1]] # 15
  - [-1, 2, C3k2, [256, False]] # 16
  
  - [-1, 1, Conv, [256, 3, 2]] # 17
  - [[-1, 13], 1, Concat, [1]] # 18
  - [-1, 2, C3k2, [512, False]] # 19
  
  - [-1, 1, Conv, [512, 3, 2]] # 20
  - [[-1, 10], 1, Concat, [1]] # 21
  - [-1, 2, C3k2, [1024, True]] # 22
  
  - [[16, 19, 22], 1, Detect, [nc]] # 23"""

# Slim Neck (GSConv for Conv, VoVGSCSP for C3k2)
neck_slim = """head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 11
  - [[-1, 6], 1, Concat, [1]] # 12
  - [-1, 2, C3Ghost, [512, False]] # 13
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 14
  - [[-1, 4], 1, Concat, [1]] # 15
  - [-1, 2, C3Ghost, [256, False]] # 16
  
  - [-1, 1, GhostConv, [256, 3, 2]] # 17
  - [[-1, 13], 1, Concat, [1]] # 18
  - [-1, 2, C3Ghost, [512, False]] # 19
  
  - [-1, 1, GhostConv, [512, 3, 2]] # 20
  - [[-1, 10], 1, Concat, [1]] # 21
  - [-1, 2, C3Ghost, [1024, True]] # 22
  
  - [[16, 19, 22], 1, Detect, [nc]] # 23"""

def build_yaml(state, has_slim_neck):
    yaml = f"# YOLO11m Ablation State {state}\n"
    yaml += "nc: 4\n"
    yaml += "scales:\n  m: [0.67, 0.75, 768]\n\n"
    yaml += "backbone:\n"
    yaml += "  - [-1, 1, Conv, [64, 3, 2]]\n"
    yaml += "  - [-1, 1, Conv, [128, 3, 2]]\n"
    yaml += "  - [-1, 2, C3k2, [128, False, 0.25]]\n"
    yaml += "  - [-1, 1, Conv, [256, 3, 2]]\n"
    yaml += "  - [-1, 2, C3k2, [256, False, 0.25]]\n"
    yaml += "  - [-1, 1, Conv, [512, 3, 2]]\n"
    yaml += "  - [-1, 2, C3k2, [512, True]]\n"
    yaml += "  - [-1, 1, Conv, [1024, 3, 2]]\n"
    yaml += "  - [-1, 2, C3k2, [1024, True]]\n"
    yaml += "  - [-1, 1, SPPF, [1024, 5]]\n"
    yaml += layer_10 + "\n\n"
    yaml += (neck_slim if has_slim_neck else neck_standard) + "\n"
    return yaml

# State mapping (has_slim_neck)
# Axis A: Slim-Neck
states = [
    (1, False), # pure baseline
    (2, True),  # A
    (3, False), # B (loss only)
    (4, False), # C (loss only)
    (5, True),  # A+B
    (6, True),  # A+C
    (7, False), # B+C
    (8, True),  # A+B+C
]

for s, has_slim in states:
    with open(f"configs/yolo11m_state{s}.yaml", "w") as f:
        f.write(build_yaml(s, has_slim))

print("Configs generated.")
