from __future__ import annotations

from ..logger import logger
import rerun as rr
import rerun.blueprint as rrb
from .rerun_wrapper_window import RerunWrapperWindow
from .rerun_wrapper_window_map import RerunWrapperWindowMap
from .rerun_wrapper_window_map_comparison import RerunWrapperWindowMapComparison
from typeguard import typechecked

@typechecked
class RerunWrapper():
    """ Wrapper for spawning and visualizing using Rerun. """

    def __init__(self, enable: bool, name: str, windows: list[RerunWrapperWindow]):
        
        # Check Window Compatibility
        if any(isinstance(w, RerunWrapperWindowMap) for w in windows) and any(isinstance(w, RerunWrapperWindowMapComparison) for w in windows):
            raise ValueError("Can't do both RerunWrapperWindowMap & RerunWrapperWindowMapComparison at the same time!")
        if any(isinstance(w, RerunWrapperWindowMapComparison) for w in windows) and len(windows) > 1:
            raise ValueError("Only single RerunWrapperWindowMapComparison window supported at a time")
        
        # Do initialization if we are enabled
        if enable:
            robot_tabs = []
            for window in windows:
                robot_tab = window._get_blueprint_part()
                robot_tabs.append(robot_tab)
            blueprint = rrb.Blueprint(rrb.Tabs(*robot_tabs))
            rr.init(name, spawn=True, default_blueprint=blueprint)

