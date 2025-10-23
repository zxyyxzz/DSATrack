import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker

import torch
import cv2
torch.set_num_threads(1)
os.environ['MKL_NUM_THREADS'] = '1'
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def run_tracker(tracker_names, tracker_params, run_ids=None, dataset_names='otb', sequence=None, debug=0, threads=0,
                num_gpus=8):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """
    print(dataset_names)
    dataset = get_dataset(dataset_names[0])

    if sequence is not None:
        dataset = [dataset[sequence]]

    # trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id)]
    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id) for tracker_name, tracker_param, dataset_name, run_id in zip(tracker_names, tracker_params, dataset_names, run_ids)]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--tracker_name', type=str, nargs='+', help='Name(s) of tracking method.', default=['grm_stu'])
    parser.add_argument('--tracker_param', type=str, nargs='+', help='Name(s) of config file.', default=['vitb_d8'])
    parser.add_argument('--runid', type=int, default=[None, None, None, None, None, None], help='The run id.', )
    parser.add_argument('--dataset_name', type=str, nargs='+', default=['dtb70'], help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=8)

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence
    
    args.runid = [i for i in range(345, 351)]
    tracker_num = len(args.runid)

    if isinstance(args.tracker_param, str):
        args.tracker_param = [args.tracker_param]
    if isinstance(args.tracker_name, str):
        args.tracker_name = [args.tracker_name]
    if isinstance(args.dataset_name, str):
        args.dataset_name = [args.dataset_name]

    args.tracker_param = args.tracker_param * tracker_num
    args.tracker_name = args.tracker_name * tracker_num
    args.dataset_name = args.dataset_name * tracker_num
    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus)


if __name__ == '__main__':
    main()
