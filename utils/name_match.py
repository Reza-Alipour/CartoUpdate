from agents.agem import AGEM
from agents.cndpm import Cndpm
from agents.ewc_pp import EWC_pp
from agents.exp_replay import ExperienceReplay
from agents.gdumb import Gdumb
from agents.icarl import Icarl
from agents.lwf import Lwf
from agents.pcr import ProxyContrastiveReplay
from agents.scr import SupContrastReplay
from continuum.dataset_scripts.cifar10 import CIFAR10
from continuum.dataset_scripts.cifar100 import CIFAR100
from continuum.dataset_scripts.core50 import CORE50
from continuum.dataset_scripts.mini_imagenet import Mini_ImageNet
from continuum.dataset_scripts.openloris import OpenLORIS
from utils.buffer.aser_retrieve import ASER_retrieve
from utils.buffer.aser_update import ASER_update
from utils.buffer.carto_update import Carto_update
from utils.buffer.gss_greedy_update import GSSGreedyUpdate
from utils.buffer.mem_match import MemMatch_retrieve
from utils.buffer.mir_retrieve import MIR_retrieve
from utils.buffer.random_retrieve import Random_retrieve
from utils.buffer.reservoir_update import Reservoir_update
from utils.buffer.sc_retrieve import Match_retrieve

data_objects = {
    'cifar100': CIFAR100,
    'cifar10': CIFAR10,
    'core50': CORE50,
    'mini_imagenet': Mini_ImageNet,
    'openloris': OpenLORIS
}

agents = {
    'ER': ExperienceReplay,
    'EWC': EWC_pp,
    'AGEM': AGEM,
    'CNDPM': Cndpm,
    'LWF': Lwf,
    'ICARL': Icarl,
    'GDUMB': Gdumb,
    'SCR': SupContrastReplay,
    'PCR': ProxyContrastiveReplay
}

retrieve_methods = {
    'MIR': MIR_retrieve,
    'random': Random_retrieve,
    'ASER': ASER_retrieve,
    'match': Match_retrieve,
    'mem_match': MemMatch_retrieve

}

update_methods = {
    'random': Reservoir_update,
    'GSS': GSSGreedyUpdate,
    'ASER': ASER_update,
    'Carto': Carto_update
}
