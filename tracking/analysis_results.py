# import matplotlib.pyplot as plt
#
# plt.rcParams['figure.figsize'] = [8, 8]
#
# import _init_paths
# from lib.test.analysis.plot_results import plot_results, print_results
# from lib.test.evaluation import get_dataset, trackerlist
#
# trackers = []
# dataset_name = 'lasot'
# resolution = 256
# for i in range(350, 351):
#
#     trackers.extend(trackerlist(name='grm', parameter_name=f'vitb_{resolution}_ep300_{i}', dataset_name=dataset_name,
#                                 run_ids=None, display_name=f'vitb_{resolution}_ep300_{i}'))
#     # trackers.extend(trackerlist(name='grm', parameter_name='vitl_320_ep300', dataset_name='NOTU',
#     #                             run_ids=None, display_name='vitl_320_ep300'))
#
#     dataset = get_dataset(dataset_name)
#     print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec'), avist=False)
# #
import _init_paths
# import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
# dataset_name = "tnl2k"
# dataset_name = 'lasot_extension_subset'
# dataset_name = 'otb'
# dataset_name = 'avist'
# dataset_name = 'lasot'
# dataset_name = 'avist'
dataset_name = 'uav123'
# dataset_name = 'uav20l'
# dataset_name = 'uav123_10fps'
# dataset_name = 'dtb70'
# dataset_name = 'visdrone'
# dataset_name = 'uavdt'
# dataset_name = 'nfs'
# dataset_name = 'lagot'
# dataset_name = 'ext'
"""stark"""
# trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-S50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline_R101', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST101'))
"""TransT"""
# trackers.extend(trackerlist(name='TransT_N2', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N2', result_only=True))
# trackers.extend(trackerlist(name='TransT_N4', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N4', result_only=True))
"""pytracking"""
# trackers.extend(trackerlist('atom', 'default', None, range(0,5), 'ATOM'))
# trackers.extend(trackerlist('dimp', 'dimp18', None, range(0,5), 'DiMP18'))
# trackers.extend(trackerlist('dimp', 'dimp50', None, range(0,5), 'DiMP50'))
# trackers.extend(trackerlist('dimp', 'prdimp18', None, range(0,5), 'PrDiMP18'))
# trackers.extend(trackerlist('dimp', 'prdimp50', None, range(0,5), 'PrDiMP50'))
"""ostrack"""
trackers.extend(trackerlist(name='grm_stu', parameter_name='vitb_d4_350', dataset_name=dataset_name,
                            run_ids=None, display_name='vitb_d4_350'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack384'))


dataset = get_dataset(dataset_name)
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
# plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
# plot_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))

