from pathlib import Path
from run_slam import run_slam

param_dir: Path =Path(__file__).parent.parent / "params" / "demo"
run_slam(str(param_dir), None, 'Sweep Debugging', None, False, None, False, False, [], False)