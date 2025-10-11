from pathlib import Path
from run_slam import run_slam

param_dir: Path = Path(__file__).parent.parent / "params" / "hercules_AustraliaEnv"
run_slam(str(param_dir), None, 'MeronomyGraph Ablation v1.1 - HERCULES', None, False, None, False, False, [], False)