import copy
from typing import List, Tuple, Optional
import torch.nn.functional as F
import einops
import torch
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks.transformer import PatchEmbed, FFN, build_transformer_layer
from mmengine.dist import is_main_process
from mmengine.model import BaseModule
from peft import get_peft_config, get_peft_model
from torch import Tensor, nn
# from mmdet.utils import OptConfigType, MultiConfig
from mmpretrain.models import resize_pos_embed
from mmpretrain.models.backbones.vit_sam import Attention, window_partition, window_unpartition
from mmseg.models import BaseSegmentor, EncoderDecoder
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmseg.utils import OptConfigType, MultiConfig
from mmpretrain.registry import MODELS
import math
from mmpretrain.models import build_norm_layer as build_norm_layer_mmpretrain

def clones(module, N):
    "工具人函数，定义N个相同的模块"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
def attention(query, key, value, mask=None, dropout=None):
    """
    实现 Scaled Dot-Product Attention
    :param query: 输入与Q矩阵相乘后的结果,size = (batch , h , L , d_model//h)
    :param key: 输入与K矩阵相乘后的结果,size同上
    :param value: 输入与V矩阵相乘后的结果，size同上
    :param mask: 掩码矩阵
    :param dropout: drop out
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  #计算QK/根号d_k，size=(batch,h,L,L)
    if mask is not None:
        # 掩码矩阵，编码器mask的size = [batch,1,1,src_L],解码器mask的size= = [batch,1,tgt_L,tgt_L]
        scores = scores.masked_fill(mask=mask, value=torch.tensor(-1e9))
    p_attn = F.softmax(scores, dim = -1)  # 以最后一个维度进行softmax(也就是最内层的行),size = (batch,h,L,L)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn  # 与V相乘。第一个输出的size为(batch,h,L,d_model//h),第二个输出的size = (batch,h,L,L)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout):
        """
        实现多头注意力机制
        :param h: 头数
        :param d_model: word embedding维度
        :param dropout: drop out
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  #检测word embedding维度是否能被h整除
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h  # 头的个数
        self.linears = clones(nn.Linear(d_model, d_model), 4) #四个线性变换，前三个为QKV三个变换矩阵，最后一个用于attention后
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.LN=nn.LayerNorm(h,eps=0,elementwise_affine=True)
        self.ffn = FFN(
            embed_dims=d_model,
            feedforward_channels=d_model*2,
            num_fcs=2,
            ffn_drop=0,
            dropout_layer=dict(type='DropPath', drop_prob=0),
            act_cfg=dict(type='GELU'))
    def forward(self, query, key, value, mask=None):
        """
        :param query: 输入x，即(word embedding+postional embedding)，size=[batch, L, d_model] tips:编解码器输入的L可能不同
        :param key: 同上，size同上
        :param value: 同上，size同上
        :param mask: 掩码矩阵，编码器mask的size = [batch , 1 , src_L],解码器mask的size = [batch, tgt_L, tgt_L]
        """
        if mask is not None:
            # 在"头"的位置增加维度，意为对所有头执行相同的mask操作
            mask = mask.unsqueeze(1)  # 编码器mask的size = [batch,1,1,src_L],解码器mask的size= = [batch,1,tgt_L,tgt_L]
        nbatches = query.size(0) # 获取batch的值，nbatches = batch

        # 1) 利用三个全连接算出QKV向量，再维度变换 [batch,L,d_model] ----> [batch , h , L , d_model//h]
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  # view中给-1可以推测这个位置的维度
             for l, x in zip(self.linears, (query, key, value))]

        # 2) 实现Scaled Dot-Product Attention。x的size = (batch,h,L,d_model//h)，attn的size = (batch,h,L,L)
        x, self.attn = attention(query, key, value, mask=mask,dropout=self.dropout)

        # 3) 这步实现拼接。transpose的结果 size = (batch , L , h , d_model//h)，view的结果size = (batch , L , d_model)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        x= self.linears[-1](x) 
        x=self.LN(x)
        x=self.ffn(x,identity=x)
        return  x# size = (batch , L , d_model)
# class SharedAttention(nn.Module):
#     def __init__(self, h, d_model, dropout):
#         """
#         实现多头注意力机制
#         :param h: 头数
#         :param d_model: word embedding维度
#         :param dropout: drop out
#         """
#         super(SharedAttention, self).__init__()
#         assert d_model % h == 0  #检测word embedding维度是否能被h整除
#         # We assume d_v always equals d_k
#         self.d_k = d_model // h
#         self.h = h  # 头的个数
#         self.linears = clones(nn.Linear(d_model, d_model), 4) #四个线性变换，前三个为QKV三个变换矩阵，最后一个用于attention后
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout)
#         self.LN=nn.LayerNorm(h,eps=0,elementwise_affine=True)
#     def forward(self, query, key, value, mask=None):
#         """
#         :param query: 输入x，即(word embedding+postional embedding)，size=[batch, L, d_model] tips:编解码器输入的L可能不同
#         :param key: 同上，size同上
#         :param value: 同上，size同上
#         :param mask: 掩码矩阵，编码器mask的size = [batch , 1 , src_L],解码器mask的size = [batch, tgt_L, tgt_L]
#         """
#         if mask is not None:
#             # 在"头"的位置增加维度，意为对所有头执行相同的mask操作
#             mask = mask.unsqueeze(1)  # 编码器mask的size = [batch,1,1,src_L],解码器mask的size= = [batch,1,tgt_L,tgt_L]
#         nbatches = query.size(0) # 获取batch的值，nbatches = batch

#         # 1) 利用三个全连接算出QKV向量，再维度变换 [batch,L,d_model] ----> [batch , h , L , d_model//h]
#         query, key, value = \
#             [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  # view中给-1可以推测这个位置的维度
#              for l, x in zip(self.linears, (query, key, value))]

#         # 2) 实现Scaled Dot-Product Attention。x的size = (batch,h,L,d_model//h)，attn的size = (batch,h,L,L)
#         x, self.attn = attention(query, key, value, mask=mask,dropout=self.dropout)

#         # 3) 这步实现拼接。transpose的结果 size = (batch , L , h , d_model//h)，view的结果size = (batch , L , d_model)
#         x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
#         x= self.linears[-1](x) 
#         x=self.LN(x)
#         return  x# size = (batch , L , d_model)
class SharedAttention(nn.Module):
    def __init__(self, h, d_model, dropout):
        """
        实现多头注意力机制
        :param h: origin channel
        :param d_model: extra channel
        :param dropout: drop out
        """
        super(SharedAttention, self).__init__()
        self.corr=nn.Bilinear(h,d_model,d_model)
        self.fc=nn.Linear(h+d_model,h)
        self.LN=nn.LayerNorm(h,eps=0,elementwise_affine=True)
    def forward(self, source, fro, ori, mask=None):
        x=torch.sigmoid(self.corr(ori,fro))
        #print(x.size())
        x=torch.mul(x,fro)
        x=torch.sigmoid(self.fc(torch.cat([source,x],dim=2)))
        x=self.LN(x)
        return  x# size = (batch , L , d_model)
@MODELS.register_module()
class MMPretrainSamVisionEncoder(BaseModule):
    def __init__(
            self,
            encoder_cfg,
            peft_cfg=None,
            init_cfg=None,
            out_dim=1408
            #embed_dims=0
    ):
        super().__init__(init_cfg=init_cfg)
        vision_encoder = MODELS.build(encoder_cfg)
        self.embed_dims=out_dim
        self.fc=nn.Linear(vision_encoder.embed_dims,out_dim)
        # self.embeddingA=nn.Parameter(torch.FloatTensor(1,out_dim),requires_grad=True)
        # self.embeddingB=nn.Parameter(torch.FloatTensor(1,out_dim),requires_grad=True)
        # self.embeddingC=nn.Parameter(torch.FloatTensor(1,out_dim),requires_grad=True)
        self.mta1=MultiHeadedAttention(h=out_dim,d_model=out_dim,dropout=0)
        self.mta2=MultiHeadedAttention(h=out_dim,d_model=out_dim,dropout=0)
        self.mta3=MultiHeadedAttention(h=out_dim,d_model=out_dim,dropout=0)
        vision_encoder.init_weights()
        if peft_cfg is not None and isinstance(peft_cfg, dict):
            config = {
                "peft_type": "LORA",
                "r": 16,
                'target_modules': ["qkv"],
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "inference_mode": False,
            }
            config.update(peft_cfg)
            peft_config = get_peft_config(config)
            self.vision_encoder = get_peft_model(vision_encoder, peft_config)
            if is_main_process():
                self.vision_encoder.print_trainable_parameters()
        else:
            self.vision_encoder = vision_encoder
            # freeze the vision encoder
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        #linear(): argument 'input' (position 1) must be Tensor, not tuple
        #我操你妈！！！
        #linear(): argument 'input' (position 1) must be Tensor, not list
        #RuntimeError: mat1 and mat2 shapes cannot be multiplied (8192x16 and 1024x1408)
        #fc的格式不对
        #print("next fuck")
        #todo：输出A,B,A-B,在DIM1上拼接
        x2,x=self.vision_encoder(x)
        x=self.fc(x)
        x=x.view(x.size(0), x.size(1) * x.size(2),x.size(3))
        x1=self.mta1(x,x,x)
        x=x+x1
        x2=self.mta2(x,x,x)
        x=x+x2
        x3=self.mta3(x,x,x)
        x=x+x3
        # print(x.size())
        # print(x.type)
        # x=x.view(x.size(0), x.size(1) * x.size(2),x.size(3))
        #A, B = torch.split(x, 1, dim=0)
        # A=A#+self.embeddingA
        # B=B#+self.embeddingB
        # C=A-B#+self.embeddingC
        # #x=torch.cat([A,B,C],dim=1)
        #x=self.mta(A,B,C)
        #print(x.size())
        return [x]
        #return self.vision_encoder(x)[-1]

@MODELS.register_module()
class MMPretrainSamVisionEncoderDual(BaseModule):
    def __init__(
            self,
            encoder_cfg,
            peft_cfg=None,
            init_cfg=None,
            out_dim=1408,
            out_dim2=256
            #embed_dims=0
    ):
        super().__init__(init_cfg=init_cfg)
        vision_encoder = MODELS.build(encoder_cfg)
        self.embed_dims=out_dim
        self.fc=nn.Linear(vision_encoder.embed_dims,out_dim)
        #self.fc2=nn.Linear(out_dim,out_dim2)

        self.mta1=MultiHeadedAttention(h=out_dim,d_model=out_dim,dropout=0)
        self.mta2=MultiHeadedAttention(h=out_dim,d_model=out_dim,dropout=0)
        #self.mta3=MultiHeadedAttention(h=out_dim,d_model=out_dim,dropout=0)
        self.sa1=SharedAttention(h=out_dim,d_model=out_dim2,dropout=0)
        self.sa2=SharedAttention(h=out_dim,d_model=out_dim2,dropout=0)
        #self.sa3=SharedAttention(h=out_dim,d_model=out_dim2,dropout=0)

        self.mta11=MultiHeadedAttention(h=out_dim2,d_model=out_dim2,dropout=0)
        self.mta22=MultiHeadedAttention(h=out_dim2,d_model=out_dim2,dropout=0)
        #self.mta33=MultiHeadedAttention(h=out_dim2,d_model=out_dim2,dropout=0)
        #self.sa11=SharedAttention(h=out_dim2,d_model=out_dim,dropout=0)
        #self.sa22=SharedAttention(h=out_dim2,d_model=out_dim,dropout=0)
        #self.sa33=SharedAttention(h=out_dim,d_model=out_dim,dropout=0)


        vision_encoder.init_weights()
        if peft_cfg is not None and isinstance(peft_cfg, dict):
            config = {
                "peft_type": "LORA",
                "r": 16,
                'target_modules': ["qkv"],
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "inference_mode": False,
            }
            config.update(peft_cfg)
            peft_config = get_peft_config(config)
            self.vision_encoder = get_peft_model(vision_encoder, peft_config)
            if is_main_process():
                self.vision_encoder.print_trainable_parameters()
        else:
            self.vision_encoder = vision_encoder
            # freeze the vision encoder
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        xp,x=self.vision_encoder(x)
        #xp=x
        #print(xp.size())
        x=self.fc(x)
        #print(xp.size())
        size=xp.size()
        x=x.view(x.size(0), x.size(1) * x.size(2),x.size(3))
        xp=xp.view(xp.size(0), xp.size(1) * xp.size(2),xp.size(3))
        #cnm!!!
        x1=self.mta1(x,x,x)
        #x1=x
        x11=self.mta11(xp,xp,xp)
        #x11=xp
        x1=self.sa1(x1,x11,x1)
        #x11=self.sa11(x11,x1,x11)
        x1=x+x1
        x11=xp+x11

        x2=self.mta2(x1,x1,x1)
        x22=self.mta22(x11,x11,x11)
        x2=self.sa2(x2,x22,x2)
        #x22=self.sa22(x22,x2,x22)
        x2=x1+x2
        x22=x11+x22

        # x3=self.mta3(x2,x2,x2)
        # x33=self.mta33(x22,x22,x22)
        # x3=self.sa3(x3,x33,x3)
        # #x33=self.sa33(x33,x3,x33)
        # x3=x2+x3
        # x33=x22+x33
        #x33=x33.view(size)
        #x33=x22.view(size)
        #x33=self.fc2(x33)
        #print(x33.size())
        # (B, H, W, C) -> (B, C, H, W)
        x22 = x22.view(size).permute(0, 3, 1, 2)
        #x11 = x11.view(size).permute(0, 3, 1, 2)
        #x33 = x33.view(size).permute(0, 3, 1, 2)
        #print(x33.size())
        return [x2,x22]
        #return [x1,x11]
        #return [x3,x33]

@MODELS.register_module()
class MLPSegHead(BaseDecodeHead):
    def __init__(
            self,
            out_size,
            interpolate_mode='bilinear',
            **kwargs
    ):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)
        self.out_size = out_size
        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=self.out_size,
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.cls_seg(out)
        return out



@MODELS.register_module()
class SequentialNeck(BaseModule):
    def __init__(self, necks):
        super().__init__()
        self.necks = nn.ModuleList()
        for neck in necks:
            self.necks.append(MODELS.build(neck))

    def forward(self, *args, **kwargs):
        for neck in self.necks:
            args = neck(*args, **kwargs)
        return args


@MODELS.register_module()
class SimpleFPN(BaseModule):
    def __init__(self,
                 backbone_channel: int,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 init_cfg: MultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.backbone_channel = backbone_channel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2),
            build_norm_layer(norm_cfg, self.backbone_channel // 2)[1],
            nn.GELU(),
            nn.ConvTranspose2d(self.backbone_channel // 2,
                               self.backbone_channel // 4, 2, 2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, input: Tensor) -> tuple:
        # build FPN
        inputs = []
        inputs.append(self.fpn1(input))
        inputs.append(self.fpn2(input))
        inputs.append(self.fpn3(input))
        inputs.append(self.fpn4(input))

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            for i in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)


@MODELS.register_module()
class TimeFusionTransformerEncoderLayer(BaseModule):
    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_fcs: int = 2,
                 qkv_bias: bool = True,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 use_rel_pos: bool = False,
                 window_size: int = 0,
                 input_size: Optional[Tuple[int, int]] = None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.window_size = window_size

        self.ln1 = build_norm_layer_mmpretrain(norm_cfg, self.embed_dims)

        self.attn = Attention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else
            (window_size, window_size),
        )

        self.ln2 = build_norm_layer_mmpretrain(norm_cfg, self.embed_dims)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

        if self.window_size > 0:
            in_channels = embed_dims * 2
            self.down_channel = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, bias=False)
            self.down_channel.weight.data.fill_(1.0/in_channels)

            self.soft_ffn = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1),
                nn.GELU(),
                nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1),
            )

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def forward(self, x):
        shortcut = x
        x = self.ln1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x

        x = self.ffn(self.ln2(x), identity=x)
        # # time phase fusion
        if self.window_size > 0:
            x = einops.rearrange(x, 'b h w d -> b d h w')  # 2B, C, H, W
            x0 = x[:x.size(0)//2]
            x1 = x[x.size(0)//2:]  # B, C, H, W
            x0_1 = torch.cat([x0, x1], dim=1)
            activate_map = self.down_channel(x0_1)
            activate_map = torch.sigmoid(activate_map)
            x0 = x0 + self.soft_ffn(x1 * activate_map)
            x1 = x1 + self.soft_ffn(x0 * activate_map)
            x = torch.cat([x0, x1], dim=0)
            x = einops.rearrange(x, 'b d h w -> b h w d')
        return x
@MODELS.register_module()
class TimeFusionTransformerEncoderLayer2(BaseModule):
    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_fcs: int = 2,
                 qkv_bias: bool = True,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 use_rel_pos: bool = False,
                 window_size: int = 0,
                 input_size: Optional[Tuple[int, int]] = None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.window_size = window_size

        self.ln1 = build_norm_layer_mmpretrain(norm_cfg, self.embed_dims)

        self.attn = Attention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else
            (window_size, window_size),
        )

        self.ln2 = build_norm_layer_mmpretrain(norm_cfg, self.embed_dims)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

        if self.window_size > 0:
            in_channels = embed_dims * 2
            self.down_channel = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, bias=False)
            self.down_channel.weight.data.fill_(1.0/in_channels)
            #input_size=input_size if window_size == 0 else (window_size, window_size)
            self.down_spatial = nn.Linear(input_size[0]*input_size[1]*2, 1)
            self.down_spatial.weight.data.fill_(1.0)
            self.soft_ffn = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1),
                nn.GELU(),
                nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1),
            )
            self.down_channel2 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, bias=False)
            self.down_channel2.weight.data.fill_(1.0/in_channels)
            #input_size=input_size if window_size == 0 else (window_size, window_size)
            self.down_spatial2 = nn.Linear(input_size[0]*input_size[1]*2, 1)
            self.down_spatial2.weight.data.fill_(1.0)
            self.soft_ffn2 = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1),
                nn.GELU(),
                nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1),
            )

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def forward(self, x):
        shortcut = x
        x = self.ln1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x

        x = self.ffn(self.ln2(x), identity=x)
        # # time phase fusion
        if self.window_size > 0:
            x = einops.rearrange(x, 'b h w d -> b d h w')  # 2B, C, H, W
            x0 = x[:x.size(0)//2]
            x1 = x[x.size(0)//2:]  # B, C, H, W
            x0_1 = torch.cat([x0, x1], dim=1)
            activate_map = self.down_channel2(x0_1)
            activate_map = torch.sigmoid(activate_map)
            #size=x1.size()
            #x1=x1.view(x1.size(0), x1.size(1), x1.size(2)*x1.size(3))
            #size=x0.size()
            #x0=x0.view(x0.size(0), x0.size(1), x0.size(2)*x0.size(3))
            #x0_1_ = torch.cat([x0, x1], dim=2)
            ##x0_1_=activate_map_.view(activate_map.size(0), activate_map.size(1), activate_map.size(2)*activate_map.size(3))
            #activate_map2 = self.down_spatial2(x0_1_)
            #activate_map2 = torch.sigmoid(activate_map2)
            #x1=activate_map2*x1
            #x1=x1.view(size)
            #x0=activate_map2*x0
            #x0=x0.view(size)


            x00 = x[:x.size(0)//2]
            x10 = x[x.size(0)//2:]  # B, C, H, W
            x0_1 = torch.cat([x00, x10], dim=1)
            #activate_map_ = self.down_channel2(x0_1)
            #activate_map_ = torch.sigmoid(activate_map_)
            size=x10.size()
            x1_=x10.view(x1.size(0), x1.size(1), x1.size(2)*x1.size(3))
            size=x00.size()
            x0_=x00.view(x0.size(0), x0.size(1), x0.size(2)*x0.size(3))
            x0_1_ = torch.cat([x0_, x1_], dim=2)
            #x0_1_=activate_map_.view(activate_map.size(0), activate_map.size(1), activate_map.size(2)*activate_map.size(3))
            activate_map2_ = self.down_spatial2(x0_1_)
            activate_map2_ = torch.sigmoid(activate_map2_)
            activate_map_ = self.down_spatial(x0_1_)
            activate_map_ = torch.sigmoid(activate_map_)
            x1_=activate_map_*x1_
            x1_=x1_.view(size)
            x0_=activate_map2_*x0_
            x0_=x0_.view(size)

            x0 = x[:x.size(0)//2] + self.soft_ffn(x0 * activate_map)+self.soft_ffn2(x0_ )
            x1 = x[x.size(0)//2:] + self.soft_ffn(x1 * activate_map)+self.soft_ffn2(x1_)
            x = torch.cat([x0, x1], dim=0)
            x = einops.rearrange(x, 'b d h w -> b h w d')
        return x
