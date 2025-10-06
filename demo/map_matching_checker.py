import numpy as np
from pathlib import Path
import pickle
from roman.object.segment import Segment  # adjust import path if needed
from roman.map.map import ROMANMap  # adjust import path if needed
import sys
from typing import List, Optional

def compare_segments(
    seg1: Segment, 
    seg2: Segment, 
    path: str = "Segment", 
    atol: float = 1e-6, 
    rtol: float = 0.0
) -> List[str]:
    diffs: List[str] = []

    # Compare IDs
    if seg1.id != seg2.id:
        diffs.append(f"{path}.id: {seg1.id} != {seg2.id}")

    # Compare points
    points1: np.ndarray = seg1.points if seg1.points is not None else np.zeros((0, 3))
    points2: np.ndarray = seg2.points if seg2.points is not None else np.zeros((0, 3))
    if points1.shape != points2.shape:
        diffs.append(f"{path}.points shape: {points1.shape} != {points2.shape}")
    elif not np.allclose(points1, points2, atol=atol, rtol=rtol, equal_nan=True):
        diffs.append(f"{path}.points values differ")

    # Compare scalar attributes
    for attr in ["volume", "first_seen", "last_seen", "num_sightings"]:
        #print(f"Checking attribute {attr}...")
        val1: float = getattr(seg1, attr)
        val2: float = getattr(seg2, attr)
        if not np.isclose(val1, val2, atol=atol, rtol=rtol):
            diffs.append(f"{path}.{attr}: {val1} != {val2}")

    # Compare function attributes
    for fun in ["linearity", "planarity", "scattering"]:
        #print(f"Checking function {fun}...")
        fun1: float = getattr(seg1, fun)
        fun2: float = getattr(seg2, fun)
        val1 = fun1()
        val2 = fun2()
        if not np.isclose(val1, val2, atol=atol, rtol=rtol):
            diffs.append(f"{path}.{attr}: {val1} != {val2}")

    # Compare vector attributes
    for attr in ["center", "extent", "semantic_descriptor"]:
        #print(f"Checking vector {attr}...")
        val1: Optional[np.ndarray] = getattr(seg1, attr)
        val2: Optional[np.ndarray] = getattr(seg2, attr)
        if val1 is None and val2 is None:
            continue
        if val1 is None or val2 is None:
            diffs.append(f"{path}.{attr}: {val1} != {val2}")
        elif not np.allclose(val1, val2, atol=atol, rtol=rtol, equal_nan=True):
            diffs.append(f"{path}.{attr} values differ")

    return diffs


def compare_roman_maps(map1: ROMANMap, map2: ROMANMap) -> List[str]:
    diffs: List[str] = []

    # Compare lengths of trajectories and times
    if len(map1.times) != len(map2.times):
        diffs.append(f"Times length mismatch: {len(map1.times)} vs {len(map2.times)}")
    if len(map1.trajectory) != len(map2.trajectory):
        diffs.append(f"Trajectory length mismatch: {len(map1.trajectory)} vs {len(map2.trajectory)}")
    
    # Compare each time
    for i, (t1, t2) in enumerate(zip(map1.times, map2.times)):
        if t1 != t2:
            diffs.append(f"Time mismatch at index {i}: {t1} vs {t2}")

    # Compare segments by ID
    seg_ids_1 = {seg.id for seg in map1.segments}
    seg_ids_2 = {seg.id for seg in map2.segments}
    print("Seg IDS 1: ", seg_ids_1)
    print("Seg IDS 2: ", seg_ids_2)
    if seg_ids_1 != seg_ids_2:
        diffs.append(f"Segment ID sets do not match: {seg_ids_1} vs {seg_ids_2}")

    for seg in map1.segments:
        seg2 = map2.get_segment_by_id(seg.id)
        if seg2 is None:
            diffs.append(f"Segment with ID {seg.id} missing in second map")
        else:
            # print(f"Comparing Segments with ID {seg.id}")
            diffs += compare_segments(seg, seg2, path=f"Segment[{seg.id}->{seg2.id}]")

    return diffs

def main() -> None:
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <map_file_1.pkl> <map_file_2.pkl> <Robot #>")
        sys.exit(1)

    map_file_1: str = Path('.').parent.parent / 'demo_output' / sys.argv[1] / 'latest' / 'map' / ('sparkal' + sys.argv[3] + '.pkl')
    map_file_2: str = Path('.').parent.parent / 'demo_output' / sys.argv[2] / 'latest' / 'map' / ('sparkal' + sys.argv[3] + '.pkl')

    # Load the ROMANMap objects
    with open(map_file_1, 'rb') as f1:
        map1: ROMANMap = pickle.load(f1)

    with open(map_file_2, 'rb') as f2:
        map2: ROMANMap = pickle.load(f2)

    # Compare the two maps
    diffs: List[str] = compare_roman_maps(map1, map2)

    if not diffs:
        print("✅ Maps are identical")
    else:
        print("❌ Maps differ:")
        for d in diffs:
            print(" -", d)


if __name__ == "__main__":
    main()