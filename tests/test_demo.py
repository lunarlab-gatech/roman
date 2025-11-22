import numpy as np
import os
from pathlib import Path
from robotdatapy.data.pose_data import PoseData
from roman.params.data_params import PoseDataGTParams, DataParams
import subprocess
import sys
from typeguard import typechecked
import unittest
import urllib.request

@typechecked
class TestDemo(unittest.TestCase):
    
    RUN_LONG_TESTS = bool(os.getenv("RUN_LONG_TESTS"))

    @unittest.skipUnless(RUN_LONG_TESTS, "Skipped because RUN_LONG_TESTS is not set")
    def test_ROMAN_matches_MeronomyGraph_disabled(self):
        """ Ensure that ROMAN and MeronomyGraph have the same output when correct parameters used."""

        # Get relevant paths
        test_dir = Path(__file__).parent.absolute()
        repo_dir = test_dir.parent
        files_dir = test_dir / 'files' / 'test_demo'
        temp_file_dir = test_dir / 'temporary_files' / 'test_demo'
        dataset_dir = temp_file_dir / 'kimera_multi_dataset'
        result_dir = temp_file_dir / 'results'
        ROMAN_result_dir = result_dir / 'ROMAN'
        MERONOMY_result_dir = result_dir / 'MERONOMY'

        # Make directory for Kimera-Multi Dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Export necessary environment variables
        os.environ["YOLO_VERBOSE"] = "False"
        os.environ["ROMAN_DEMO_DATA"] = str(dataset_dir)
        os.environ["ROMAN_WEIGHTS"] = str(repo_dir / "weights")

        # Download the dataset from Dropbox
        def reporthook(block_num: int, block_size: int, total_size: int):
            downloaded = block_num * block_size
            percent = downloaded / total_size * 100 if total_size > 0 else 0
            sys.stdout.write(f"\rDownloading... {percent:.2f}% ({downloaded / 1e6:.2f} MB / {total_size / 1e6:.2f} MB)")
            sys.stdout.flush()

        if not (dataset_dir / "sparkal1_camera.bag").exists():
            urllib.request.urlretrieve("https://www.dropbox.com/scl/fi/yn4rws8zjxbwwshlaktcm/sparkal1_camera.bag?rlkey=9yssgxg7dg2ci4oyy56nz4l9u&st=17zptsdp&dl=1", dataset_dir / "sparkal1_camera.bag", reporthook)
        if not (dataset_dir / "sparkal1_gt.csv").exists():
            urllib.request.urlretrieve("https://www.dropbox.com/scl/fi/fjppjfg66qg24pxy8o2st/sparkal1_gt.csv?rlkey=yzm70zqezghcqhl3c26brrd45&st=ono9xcr5&dl=1", dataset_dir / "sparkal1_gt.csv", reporthook)
        if not (dataset_dir / "sparkal1_vio.bag").exists():
            urllib.request.urlretrieve("https://www.dropbox.com/scl/fi/37iuugdcm75jbpn44eahn/sparkal1_vio.bag?rlkey=ts1pj8usk1audns6lq19gjvnk&st=c6dcn36y&dl=1", dataset_dir / "sparkal1_vio.bag", reporthook)

        if not (dataset_dir / "sparkal2_camera.bag").exists():
            urllib.request.urlretrieve("https://www.dropbox.com/scl/fi/nniqyll84tn96lejsgkff/sparkal2_camera.bag?rlkey=w56tpa3nkxc0yyrr1nnmj93k1&st=uxy3lfcm&dl=1", dataset_dir / "sparkal2_camera.bag", reporthook)
        if not (dataset_dir / "sparkal2_gt.csv").exists():
            urllib.request.urlretrieve("https://www.dropbox.com/scl/fi/q8z44v1uevp4f7yirk1oi/sparkal2_gt.csv?rlkey=l3pj16f4d9bf8z2z2jcy976x4&st=3dzi3ctt&dl=1", dataset_dir / "sparkal2_gt.csv", reporthook)
        if not (dataset_dir / "sparkal2_vio.bag").exists():
            urllib.request.urlretrieve("https://www.dropbox.com/scl/fi/wdtulm31ubnmiu3sk5016/sparkal2_vio.bag?rlkey=xm9yiuke7csebvfqylpt4uwdd&st=ejhnwhkk&dl=1", dataset_dir / "sparkal2_vio.bag", reporthook)

        # Run ROMAN and MeronomyGraph Disabled
        subprocess.run(['python3', repo_dir / 'research' / 'run_slam.py', '-p', files_dir / 'demo_ROMAN', '--disable-wandb', '--output-dir', ROMAN_result_dir], check=True)
        subprocess.run(['python3', repo_dir / 'research' / 'run_slam.py', '-p', files_dir / 'demo_MERONOMY', '--disable-wandb', '--output-dir', MERONOMY_result_dir], check=True)

        # Load the two RMS ATE and assert they are the same
        result_ROMAN = float((ROMAN_result_dir / 'offline_rpgo' / 'ate_rmse.txt').read_text().strip())
        result_MERONOMY = float((MERONOMY_result_dir / 'offline_rpgo' / 'ate_rmse.txt').read_text().strip())
        np.testing.assert_equal(result_ROMAN, result_MERONOMY)

if __name__ == "__main__":
    unittest.main()