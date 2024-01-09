from .DukeMTMCReid import DukeMTMCreID
from .Market1501 import Market1501
from .LTCC import LTCC
from .LTCC_ori import LTCC_ORI
from .VC_Clothes import VC_clothes
from .PRCC import PRCC
from .DP3D import DP3D
__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'ltcc':LTCC,
    'ltcc_ori':LTCC_ORI,
    'vc_clothes':VC_clothes,
    'prcc':PRCC,
    'DP3D':DP3D,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)




