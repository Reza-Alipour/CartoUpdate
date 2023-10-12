from copy import deepcopy

import torch

from utils import name_match
from utils.buffer.buffer_utils import BufferClassTracker
from utils.setup_elements import input_size_match
from utils.setup_elements import n_classes
from utils.utils import maybe_cuda


class Buffer(torch.nn.Module):
    def __init__(self, model, params):
        super().__init__()
        self.params = params
        self.model = model
        self.cuda = self.params.cuda
        self.current_index = 0
        self.n_seen_so_far = 0
        self.device = "cuda" if self.params.cuda else "cpu"

        # define buffer
        buffer_size = params.mem_size
        print('buffer has %d slots' % buffer_size)
        input_size = input_size_match[params.data]
        buffer_img = maybe_cuda(torch.FloatTensor(buffer_size, *input_size).fill_(0))
        buffer_label = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))

        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('buffer_img', buffer_img)
        self.register_buffer('buffer_label', buffer_label)
        self.carto_buffer = params.update == 'Carto'
        if params.update == 'Carto':
            params.mem_size = params.mem_size // 2
            params_cp = deepcopy(params)
            params_cp.update = 'random'
            params_cp.mem_size = params_cp.mem_size
            self.sub_buffer = Buffer(model, params_cp)
            self.eps_mem_batch = params.eps_mem_batch

        # define update and retrieve method
        self.update_method = name_match.update_methods[params.update](params)
        self.retrieve_method = name_match.retrieve_methods[params.retrieve](params)

        if self.params.buffer_tracker:
            self.buffer_tracker = BufferClassTracker(n_classes[params.data], self.device)

    def update(self, x, y, **kwargs):
        if kwargs.pop('instant_memory', None):
            return self.sub_buffer.update(x, y, **kwargs)
        return self.update_method.update(buffer=self, x=x, y=y, **kwargs)

    def retrieve(self, **kwargs):
        if self.carto_buffer:
            instant_buf_x, instant_buf_y = self.sub_buffer.retrieve(**kwargs)
            carto_buf_x, carto_buf_y = self.retrieve_method.retrieve(buffer=self, **kwargs)
            x = torch.cat([instant_buf_x, carto_buf_x], dim=0)
            y = torch.cat([instant_buf_y, carto_buf_y], dim=0)
            num_samples = x.shape[0]
            indices = torch.randperm(num_samples)
            to_select = min(self.eps_mem_batch, num_samples)
            chosen_indices = indices[:to_select]
            return x[chosen_indices], y[chosen_indices]

        return self.retrieve_method.retrieve(buffer=self, **kwargs)
