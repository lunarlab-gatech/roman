from pathlib import Path
from run_slam import run_slam

param_dir: Path = Path(__file__).parent.parent / "params" / "hercules_AustraliaEnv"
run_slam(str(param_dir), None, 'HERCULES Experiment 1.0 - ROMAN Baseline', None, False, None, False, False, False)