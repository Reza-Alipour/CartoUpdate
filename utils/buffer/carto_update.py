from itertools import groupby

import torch


class Carto_update(object):
    def __init__(self, params):
        super().__init__()
        self.samples = []

    def update(self, buffer, x, y, **kwargs):
        task_seen = kwargs['task_seen']
        carto_data = kwargs['carto_data']
        classes_seen = kwargs['classes_seen']
        samples_per_class = buffer.buffer_img.size(0) // len(classes_seen)

        pruned_samples = []
        new_samples = []
        new_image_buf = None
        new_label_buf = None
        for l, samples in groupby(sorted(self.samples, key=lambda s: s.label), key=lambda s: s.label):
            samples = list(samples)
            samples.sort(key=lambda s: s.get_variability(), reverse=False)
            pruned_samples += samples[:samples_per_class]

        new_task_samples = list(carto_data[task_seen].values())
        for l, samples in groupby(sorted(new_task_samples, key=lambda s: s.label), key=lambda s: s.label):
            samples = list(samples)
            samples.sort(key=lambda s: s.get_variability(), reverse=True)
            samples = samples[:max(len(samples) // 3, samples_per_class)]
            if len(samples) > samples_per_class:
                samples = samples[-samples_per_class:]
            new_samples += samples
        self.samples = pruned_samples + new_samples

        for i, s in enumerate(pruned_samples):
            if new_image_buf is None:
                new_image_buf = buffer.buffer_img[s.buffer_index].unsqueeze(0).to(buffer.device)
                new_label_buf = buffer.buffer_label[s.buffer_index].unsqueeze(0).to(buffer.device)
            else:
                new_image_buf = torch.cat(
                    (new_image_buf, buffer.buffer_img[s.buffer_index].unsqueeze(0).to(buffer.device)))
                new_label_buf = torch.cat(
                    (new_label_buf, buffer.buffer_label[s.buffer_index].unsqueeze(0).to(buffer.device)))
            s.buffer_index = i

        for i, s in enumerate(new_samples):
            if new_image_buf is None:
                new_image_buf = x[s.index].unsqueeze(0).to(buffer.device)
                new_label_buf = y[s.index].unsqueeze(0).to(buffer.device)
            else:
                new_image_buf = torch.cat((new_image_buf, x[s.index].unsqueeze(0).to(buffer.device)))
                new_label_buf = torch.cat((new_label_buf, y[s.index].unsqueeze(0).to(buffer.device)))
            s.buffer_index = i + len(pruned_samples)

        buffer.buffer_img[:new_image_buf.size(0)] = new_image_buf
        buffer.buffer_label[:new_label_buf.size(0)] = new_label_buf
        buffer.current_index = new_image_buf.size(0)
