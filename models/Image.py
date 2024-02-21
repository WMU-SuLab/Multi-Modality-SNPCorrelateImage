# -*- encoding: utf-8 -*-
"""
@File Name      :   image.py   
@Create Time    :   2024/1/22 15:11
@Description    :  
@Version        :  
@License        :  
@Author         :   diklios
@Contact Email  :   diklios5768@gmail.com
@Github         :   https://github.com/diklios5768
@Blog           :  
@Motto          :   All our science, measured against reality, is primitive and childlike - and yet it is the most precious thing we have.
@Other Info     :
"""
__auth__ = 'diklios'

from timm.models.layers import trunc_normal_
from torch import nn

from .ConvNeXt.with_attention import convnext_tiny
from .RETFound import vit_large_patch16


class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=1):
        super(ConvNeXtTiny, self).__init__()
        # 这个模型已经把最后一层分类去掉了，留下的是特征层
        self.image_features = convnext_tiny(num_classes)
        # self.image_features = convnext_base(1)
        self.image_mlp = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 1),
        )
        # 模型参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def load_image_model_state_dict(self, checkpoint_model):
        return self.image_features.load_state_dict(state_dict=checkpoint_model, strict=False)

    def forward(self, image):
        x = self.image_features(image)
        y = self.image_mlp(x)
        return y


class RETFoundNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vit_large_patch16(
            num_classes=1,
            drop_path_rate=0.2,
            global_pool=False,
        )

    def load_image_model_state_dict(self, checkpoint_model):
        state_dict = self.model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.model.load_state_dict(checkpoint_model, strict=False)

        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
        # manually initialize fc layer
        trunc_normal_(self.model.head.weight, std=2e-5)
        # print("Model = %s" % str(self.model))
        return msg

    def forward(self, image):
        y = self.model(image)
        return y
