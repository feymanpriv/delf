from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

class AttrDict(dict):
    """AttrDict"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value

cfg = AttrDict()

cfg.model_path = "./r101delg/"
cfg.image_scales = "0.5, 0.7071, 1.0, 1.4142, 2.0"
cfg.use_global_features = True
cfg.iou_threshold = 1.0
cfg.max_feature_num = 1000
cfg.score_threshold = 175.0
