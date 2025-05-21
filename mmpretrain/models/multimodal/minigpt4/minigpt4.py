# Copyright (c) OpenMMLab. All rights reserved.
import random
import math
import re
from typing import List, Optional, Tuple
import random
import torch
import torch.nn as nn
from mmengine.logging import MMLogger
from mmengine.model import BaseModel
from peft import get_peft_config, get_peft_model
from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample
import copy
import copy
from typing import List, Tuple, Optional
import torch.nn.functional as F
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

        return self.linears[-1](x)   # size = (batch , L , d_model)
@MODELS.register_module()
class MiniGPT4(BaseModel):
    """The multi-modality model of MiniGPT-4.

    The implementation of `MiniGPT-4 <https://arxiv.org/abs/2304.10592>`_.
    Modified from https://github.com/Vision-CAIR/MiniGPT-4/blob/main/minigpt4/models/mini_gpt4.py

    Args:
        vision_encoder (dict): The config for vision encoder.
        q_former_model (dict): The config for Qformer.
        lang_encoder (dict): The config for language model.
        tokenizer (dict): The config for tokenizer.
        task (str): To define the task, which control the processing of text.
            Defaults to 'caption'.
        freeze_vit (bool): Freeze the training of ViT. Defaults to True.
        freeze_q_former (bool): Freeze the training of Qformer. Defaults to
            True.
        num_query_token (int): Number of query tokens of Qformer. Defaults to
            32.
        prompt_template (dict): Multi-language prompt template of the model. Defaults to dict([ ('en', '###Ask: {} ###Answer: '),
                                                                                                ('zh', '###问：{} ###答：')])
        raw_prompts (dict): Prompts for training. Defaults to dict().
        max_txt_len (int): Max token length while doing tokenization. Defaults
            to 32.
        end_sym (str): Ended symbol of the sequence. Defaults to '###'.
        generation_cfg (dict): The config of text generation. Defaults to
            dict().
        data_preprocessor (:obj:`BaseDataPreprocessor`): Used for
            pre-processing data sampled by dataloader to the format accepted by
            :meth:`forward`. Defaults to None.
        init_cfg (dict): Initialization config dict. Defaults to None.
    """ # noqa

    def __init__(self,
                 vision_encoder: dict,
                 q_former_model: dict,
                 lang_encoder: dict,
                 tokenizer: dict,
                 task: str = 'caption',
                 peft_cfg=None,
                 freeze_vit: bool = True,
                 freeze_q_former: bool = True,
                 num_query_token: int = 32,
                 prompt_template: dict = dict([('en',
                                                '###Ask: {} ###Answer: '),
                                               ('zh', '###问：{} ###答：')]),
                 raw_prompts: dict = dict(),
                 max_txt_len: int = 32,
                 end_sym: str = '###',
                 generation_cfg: dict = dict(),
                 data_preprocessor: Optional[dict] = None,
                 backbone_inchannels:int=3,
                 init_cfg: Optional[dict] = None):
        if data_preprocessor is None:
            data_preprocessor = {}
        data_preprocessor.setdefault('type', 'MultiModalDataPreprocessor')
        data_preprocessor = MODELS.build(data_preprocessor)

        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.task = task
        logger = MMLogger.get_current_instance()
        self.backbone_inchannels=backbone_inchannels
        # build vision model
        vision_encoder_weight = vision_encoder.pop('pretrained', None)
        self.vision_encoder = MODELS.build(vision_encoder)
        self.ln_vision = nn.LayerNorm(self.vision_encoder.embed_dims)

        #self.mta=MultiHeadedAttention(h=4096,d_model=4096,dropout=0)
        self.fc1=torch.nn.Linear( 2*4096,4096)
        self.fc2=torch.nn.Linear( 2*4096,4096)

        if vision_encoder_weight is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(self.vision_encoder, vision_encoder_weight)
            self.vision_encoder.is_init = True
        if freeze_vit:
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
        else:
            logger.warning('Please check `frozen_stages` in the dict of'
                           '`vision_encoder`. Also set it to be -1 if do not'
                           'freeze ViT.')

        # build Qformer
        q_former_model_weight = q_former_model.pop('pretrained', None)
        self.q_former = MODELS.build(q_former_model)
        self.q_former.cls = None
        
        self.q_former.bert.embeddings.word_embeddings = None
        self.q_former.bert.embeddings.position_embeddings = None
        for layer in self.q_former.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.q_former.config.hidden_size))
        self.query_tokens.data.normal_(
            mean=0.0, std=self.q_former.config.initializer_range)

        if q_former_model_weight is not None:
            from mmengine.runner.checkpoint import CheckpointLoader
            state_dict = CheckpointLoader.load_checkpoint(
                q_former_model_weight)['state_dict']
            self.load_state_dict(state_dict, strict=False)
            # The ln_vision weights are also in the q-former checkpoint.
            setattr(self.ln_vision, 'is_init', True)
            setattr(self.q_former, 'is_init', True)

        if freeze_q_former:
            for name, param in self.q_former.named_parameters():
                param.requires_grad = False
            self.q_former.eval()
            self.query_tokens.requires_grad = False

        # build language model
        self.llama_tokenizer = TOKENIZER.build(tokenizer)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        #print(self.llama_tokenizer)
        self.llama_model = MODELS.build(lang_encoder)
        #print(self.llama_model)
        if peft_cfg is not None and isinstance(peft_cfg, dict):
            config = {
                "peft_type": "LORA",
                "r": 16,
                'target_modules': [
                "q_proj",
                "v_proj",
                "k_proj"
            ],
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "inference_mode": False,
            }
            config.update(peft_cfg)
            peft_config = get_peft_config(config)
            self.llama_model = get_peft_model(self.llama_model, peft_config)
#             config = {
#                 "peft_type": "LORA",
#                 "r": 16,
#                 'target_modules': [
# "q_proj"
# ],
#                 "lora_alpha": 32,
#                 "lora_dropout": 0.05,
#                 "bias": "none",
#                 "inference_mode": False,
#             }
#             #config.update(peft_cfg)
#             peft_config = get_peft_config(config)
#             self.q_former=get_peft_model(self.q_former,peft_config)
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        
        # build linear projection layer
        self.llama_proj = nn.Linear(self.q_former.config.hidden_size,
                                    self.llama_model.config.hidden_size)
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.end_token_id = self.llama_tokenizer.encode(end_sym)[-1]

        # set prompts
        self.en_prompt_list, self.zh_prompt_list = [], []
        if raw_prompts.get('en') is not None:
            en_filted_prompts = [
                raw_prompt for raw_prompt in raw_prompts['en']
                if '<ImageHere>' in raw_prompt
            ]
            self.en_prompt_list = [
                prompt_template['en'].format(p) for p in en_filted_prompts
            ]
        if raw_prompts.get('zh') is not None:
            zh_filted_prompts = [
                raw_prompt for raw_prompt in raw_prompts['zh']
                if '<ImageHere>' in raw_prompt
            ]
            self.zh_prompt_list = [
                prompt_template['zh'].format(p) for p in zh_filted_prompts
            ]

        # update generation configs
        self.generation_cfg = dict(
            max_new_tokens=300,
            num_beams=1,
            do_sample=True,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.1,
            length_penalty=1.0,
            temperature=1.0)
        self.generation_cfg.update(**generation_cfg)

        if hasattr(self, 'register_load_state_dict_post_hook'):
            self.register_load_state_dict_post_hook(self._load_llama_proj_hook)

    def half(self):
        self.llama_model = self.llama_model.half()
        return self

    def encode_img(self,
                   inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The function to encode the inputs."""
        device = inputs.device
        #双输入
        #print(inputs.size())
        img_from, img_to = torch.split(inputs, self.backbone_inchannels, dim=1)
        inputs = torch.cat([img_from, img_to], dim=0)
        x = self.vision_encoder(inputs)[0]
        #print(x.size())
        # #强制使得prompt与image_embeds维度匹配,其实应该删两prompt去的
        # x=x.repeat(2)
        image_embeds = self.ln_vision(x).to(device)
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(device)


        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.q_former.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        #print(query_output.last_hidden_state.size())
        inputs_llama = self.llama_proj(query_output.last_hidden_state)
        A, B = torch.split(inputs_llama, 1, dim=0)
        #print(A.size(),B.size())
        # A=A#+self.embeddingA
        # B=B#+self.embeddingB
        # C=A-B#+self.embeddingC
        #x=torch.cat([A,B,C],dim=1)
        #inputs_llama=torch.cat([A,B,C],dim=1)
        #A=A.view(A.size(0),4,8,A.size(2))#+self.embeddingA
        #B=B.view(B.size(0),4,8,B.size(2))#+self.embeddingB
        #C=A-B#+self.embeddingC
        #inputs_llama=self.mta(A,B,C)
        #print(x.size())
        #inputs_llama=x.view(x.size(0), x.size(1) * x.size(2),x.size(3))
        # print('fuck')
        # print(inputs_llama.size())
        AA=torch.sigmoid(self.fc1(torch.cat([A,B],dim=2)))
        BB=torch.sigmoid(self.fc2(torch.cat([B,A],dim=2)))
        A=torch.mul(A,AA)
        B=torch.mul(B,BB)
        inputs_llama=torch.cat([A,B,A-B],dim=1)
        atts_llama = torch.ones(
            inputs_llama.size()[:-1], dtype=torch.long).to(inputs.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds: torch.Tensor, atts_img: torch.Tensor,
                    prompt: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """The function to wrap the image and prompt.

        ########Make sure that len(prompt) == img_embeds.shape[0].#

        Args:
            img_embeds (torch.Tensor): The embedding of the input inputs.
            atts_img (torch.Tensor): Attention map of the image embeddings.
            prompt (List[str]): The prompt of the batch data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The embedding and attention map.
        """
        if len(prompt) > 0:
            while(len(prompt)>img_embeds.shape[0]):
                  del prompt[random.randint(0,len(prompt)-1)]
            p_before_list, p_after_list = [], []
            for pro in prompt:
                p_before, p_after = pro.split('<ImageHere>')
                p_before_list.append(p_before)
                p_after_list.append(p_after)
            p_before_tokens = self.llama_tokenizer(
                p_before_list,
                return_tensors='pt',
                padding='longest',
                add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after_list,
                return_tensors='pt',
                padding='longest',
                add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.model.embed_tokens(
                p_before_tokens.input_ids)
            p_after_embeds = self.llama_model.model.model.embed_tokens(
                p_after_tokens.input_ids)
            # print(p_before_embeds.size())
            # print(img_embeds.size())
            # print(p_after_embeds.size())
            wrapped_img_embeds = torch.cat(
                [p_before_embeds, img_embeds, p_after_embeds], dim=1)
            #print(wrapped_img_embeds.size())
            wrapped_atts_img = atts_img[:, :1].expand(
                -1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def loss(self,
             inputs: torch.Tensor,
             data_samples: Optional[List[DataSample]] = None) -> dict:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input inputs.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        img_embeds, atts_img = self.encode_img(inputs)

        self.llama_tokenizer.padding_side = 'right'

        prompts, texts = [], []
        # for t in data_samples:
        #     chat_content = t.chat_content
        #     split_mark = '###Answer: ' if t.lang == 'en' else '###答：'
        #     prompt, text = chat_content.split(split_mark)
        #     prompt += split_mark
        #     text += self.end_sym
        #     prompts.append(prompt)
        #     texts.append(text)
        #print(self.en_prompt_list)
        for p in self.en_prompt_list:
            split_mark = '###Answer: '
            prompt, text = p.split(split_mark)
            #prompt=prompt.replace("<Img><ImageHere></Img> ",'')
            prompt += split_mark
            prompts.append(prompt)
        #print(prompts)
        for t in data_samples:
            #print(t.gt_caption)
            text=t.gt_caption
            text[0] += self.end_sym
            #print(text)
            texts.append(text[0])
            #texts.append(text[0])
            #texts.append(tt)
            #t.chat_content
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompts)
        #print(texts)
        to_regress_tokens = self.llama_tokenizer(
            texts,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False).to(inputs.device)
        #print(to_regress_tokens)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id,
            -100)

        empty_targets = (
            torch.ones([targets.shape[0], atts_img.shape[1] + 1],
                       dtype=torch.long).to(inputs.device).fill_(
                           -100)  # plus one for bos
        )
        #print(empty_targets.size())
        #print(targets.size())
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device
                         ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.llama_model.model.model.embed_tokens(
            to_regress_tokens.input_ids)
        # print(bos_embeds.size())
        # print(img_embeds.size())
        # print(to_regress_embeds.size())
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds],
                                  dim=1)
        attention_mask = torch.cat(
            [atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return dict(loss=loss)

    def predict(
            self,
            inputs: torch.Tensor,
            data_samples: Optional[List[DataSample]] = None
    ) -> List[DataSample]:

        with torch.no_grad():
            img_embeds, atts_img = self.encode_img(inputs)

        # prompts = [
        #     random.choice(self.zh_prompt_list) if hasattr(t, 'lang')
        #     and t.lang == 'zh' else random.choice(self.en_prompt_list)
        #     for t in data_samples
        # ]
        prompts=[]
        for p in self.en_prompt_list:
            split_mark = '###Answer: '
            prompt, text = p.split(split_mark)
            #prompt=prompt.replace("<Img><ImageHere></Img> ",'')
            prompt += split_mark
            prompts.append(prompt)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompts)

        batch_size = img_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1], dtype=torch.long,
            device=img_embeds.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.model.embed_tokens(bos)
        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            eos_token_id=self.end_token_id,
            **self.generation_cfg)

        return self.post_process(outputs, data_samples)

    def post_process(
            self, outputs: torch.Tensor,
            data_samples: Optional[List[DataSample]]) -> List[DataSample]:
        """Perform post process for outputs for different task.

        Args:
            outputs (torch.Tensor): The generated outputs.
            data_samples (List[DataSample], optional): The annotation
                data of every samples.

        Returns:
            List[DataSample]: Return list of data samples.
        """
        outputs = self.llama_tokenizer.batch_decode(
            outputs, skip_special_tokens=True)

        if data_samples is None:
            data_samples = [DataSample() for _ in range(len(outputs))]

        for output, data_sample in zip(outputs, data_samples):
            if self.task == 'caption':
                output = output.split(self.end_sym)[0]
                data_sample.pred_caption = output
            else:
                # raw output
                data_sample.pred_output = output
        return data_samples

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[list] = None,
        mode: str = 'predict',
        **kwargs,
    ):
        """The unified entry for a forward process in both training and test.
        The method accepts the following modes:

        - "predict": Forward and return a list of data samples contain the
          predict results.

        Args:
            inputs (torch.Tensor): the preprocessed image tensor of shape
                ``(N, C, H, W)``.
            data_samples (List[DataSample], optional): The annotation data
                of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'predict'.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    @staticmethod
    def _load_llama_proj_hook(module, incompatible_keys):
        """Avoid warning missing keys except LLaMA projection keys."""
        proj_patterns = [
            'vision_encoder.*',
            'ln_vision.*',
            'q_former.*',
            'query_tokens',
            'llama_model.*',
        ]
        for key in list(incompatible_keys.missing_keys):
            if any(re.match(pattern, key) for pattern in proj_patterns):
                incompatible_keys.missing_keys.remove(key)
