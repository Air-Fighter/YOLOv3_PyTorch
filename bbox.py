from __future__ import division

import torch

def confidence_filter(result, confidence):
    conf_mask = (result[:, :, 4] > confidence).float().unsqueeze(2)
    result = result * conf_mask
    return result

def confidence_filter_cls(result, confidence):
    max_scores = torch.max(result[:, :, 5:25], 2)[0]
    res = torch.cat((result, max_scores), 2)
    print(res.shape)

    cond_1 = (res[:, :, 4] > confidence).float()
    cond_2 = (res[:, :, 25] > 0.995).float()

