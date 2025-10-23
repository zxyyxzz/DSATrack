import importlib
from collections import namedtuple

from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = 'lib.test.evaluation.%sdataset'  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    otb=DatasetInfo(module=pt % 'otb', class_name='OTBDataset', kwargs=dict()),
    nfs=DatasetInfo(module=pt % 'nfs', class_name='NFSDataset', kwargs=dict()),
    uav123_10fps=DatasetInfo(module=pt % "uav123_10fps", class_name="UAV123_10fpsDataset", kwargs=dict()),
    uav123=DatasetInfo(module=pt % 'uav123', class_name='UAV123Dataset', kwargs=dict()),
    uav20l=DatasetInfo(module=pt % "uav20l", class_name="UAV20LDataset", kwargs=dict()),
    dtb70=DatasetInfo(module=pt % "dtb70", class_name="DTB70Dataset", kwargs=dict()),
    visdrone=DatasetInfo(module=pt % "visdrone", class_name="VISDRONEDataset", kwargs=dict()),
    uavdt=DatasetInfo(module=pt % "uavdt", class_name="UAVDTDataset", kwargs=dict()),
    trackingnet=DatasetInfo(module=pt % 'trackingnet', class_name='TrackingNetDataset', kwargs=dict()),
    got10k_test=DatasetInfo(module=pt % 'got10k', class_name='GOT10KDataset', kwargs=dict(split='test')),
    got10k_val=DatasetInfo(module=pt % 'got10k', class_name='GOT10KDataset', kwargs=dict(split='val')),
    lasot=DatasetInfo(module=pt % 'lasot', class_name='LaSOTDataset', kwargs=dict(subset='testing')),
    lasot_ext=DatasetInfo(module=pt % 'lasot', class_name='LaSOTDataset', kwargs=dict(subset='extension')),
    itb=DatasetInfo(module=pt % 'itb', class_name='ITBDataset', kwargs=dict()),
    tnl2k=DatasetInfo(module=pt % 'tnl2k', class_name='TNL2kDataset', kwargs=dict()),
    avist=DatasetInfo(module=pt % 'avist', class_name='AVisTDataset', kwargs=dict()),
    lagot=DatasetInfo(module=pt % 'lagot', class_name='LaGOTDataset', kwargs=dict(sot_mode=True)),
    ext=DatasetInfo(module=pt % 'ext', class_name='EXTDataset', kwargs=dict()),
)


def load_dataset(name: str):
    """
    Import and load a single dataset.
    """

    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('ERROR: unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  # Call the constructor
    return dataset.get_sequence_list()


def get_dataset(*args):
    """
    Get a single or set of datasets.
    """

    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset
