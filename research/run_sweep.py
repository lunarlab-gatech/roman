from pathlib import Path
from run_slam import run_slam

param_dir: Path = Path(__file__).parent.parent / "params" / "hercules_AustraliaEnv"
run_slam(str(param_dir), None, 'ROMAN - HERCULES - V1.1', None, False, None, False, False, False)