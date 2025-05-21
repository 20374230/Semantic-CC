
# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import tempfile
from typing import List, Optional
import numpy as np
import mmengine
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend
from mmengine.evaluator import BaseMetric
import cv2
from mmengine.utils import track_iter_progress
# from nltk.translate.meteor_score import meteor_score as mt
# from nltk.translate.bleu_score import sentence_bleu
# #from nltk.corpus import cider
# from rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
#from .eval_func.cider.cider import Cider
from mmpretrain.registry import METRICS
from mmpretrain.utils import require
#from .utils import *
from pycocoevalcap.cider.cider import Cider
from .trail import Scorer
from collections import OrderedDict
from typing import Optional, Sequence, Dict
import numpy as np
import torch
from mmengine import MMLogger, print_log
from mmengine.evaluator import BaseMetric
from prettytable import PrettyTable
from torchmetrics.functional.classification import multiclass_precision, multiclass_recall, multiclass_f1_score, \
	multiclass_jaccard_index, multiclass_accuracy, binary_accuracy
from torchmetrics.functional.classification import multiclass_precision, multiclass_recall, multiclass_f1_score, \
	multiclass_jaccard_index, multiclass_accuracy, binary_accuracy
try:
    from pycocoevalcap.eval import COCOEvalCap
    from pycocotools.coco import COCO
except ImportError:
    COCOEvalCap = None
    COCO = None

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
@METRICS.register_module()
class LevirCCcaption(BaseMetric):
    """Coco Caption evaluation wrapper.

    Save the generated captions and transform into coco format.
    Calling COCO API for caption metrics.

    Args:
        ann_file (str): the path for the COCO format caption ground truth
            json file, load for evaluations.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Should be modified according to the
            `retrieval_type` for unambiguous results. Defaults to TR.
    """

    @require('pycocoevalcap')
    def __init__(self,
                 ann_file: str,
                 txt_path:str,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file
        self.txt_path=txt_path
        self.rouge=Rouge()
        self.scorer=Scorer()
        self.cider_scorer = Cider(n=4, sigma=6.0)
        with open(self.txt_path,'w') as f:
            f.write('cnm')
    def process(self, data_batch, data_samples):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        with open(self.txt_path,'a') as f:
            #f.write(data_samples)
            for data_sample in data_samples:
                result = dict()
                #print(data_sample)
                f.write(json.dumps(data_sample,cls=NpEncoder)+'\n')
                result['caption'] = data_sample.get('pred_caption')
                result['image_id'] = int(data_sample.get('image_id'))
                result['gt_caption']=data_sample.get('gt_caption')
				# #print(result)
				# # Save the result to `self.results`.
                self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method.

        # with tempfile.TemporaryDirectory() as temp_dir:

        #     eval_result_file = save_result(
        #         result=results,
        #         result_dir=temp_dir,
        #         filename='m4-caption_pred',
        #         remove_duplicate='image_id',
        #     )

        #     coco_val = coco_caption_eval(eval_result_file, self.ann_file)

        # return coco_val
        #print(results)
        crrid=0
        gt_dict=dict()
        ref_dict=dict()
        t=0
        # for i in range(0,len(results)-1):
        #     res=results[i]
            
        #     gt=[res['gt_caption'][0]]
        #     gt_dict[str(i)]=gt
        #     best=[0,0,0,0,0,0]
        #     ref=[]
        #     for j in range(0,5):
        #         crr=results[i-j]['caption']
        #         ref.append(crr)
        #     ref_dict[str(i)]=ref
        #     print(gt_dict)
        #     print(ref_dict)
        #     t+=1 
        #     if crrid!=res['image_id']:
        #         crrid=res['image_id']
        #         t=0
        p=0
        for i in range(0,len(results)):

            if crrid!=results[i]['image_id']:
                crrid=results[i]['image_id']
                t=0
            else:
                t+=1
                continue
            #best=[0,0,0,0,0,0]
            gt=[]
            for j in range(0,5):
                #print(i+j-t)
                crr=results[i+j-t]['gt_caption'][0]
                gt.append(crr)
            res=results[i]
            #print(i)
            #print(results)
            ref=[res['caption']]
            ref_dict[str(p)]=ref
            gt_dict[str(p)]=gt
            #print(gt_dict[str(i)])
            #print(ref_dict[str(i)])
            p+=1
            t+=1 
     
                #break
        score=self.scorer.compute_scores(ref_dict,gt_dict)
        #best=self.get_best(score,best)
        print(score)
        return score

    # def cal_score(self,gt,pred):
    #     #'list' object has no attribute 'split'
    #     # print(gt,pred)
    #     # meteor_score=round(mt([gt.split()],pred.split()),4)
    #     # bleu_score_1=sentence_bleu(gt,pred,weights=(1,0,0,0))
    #     # bleu_score_2=sentence_bleu(gt,pred,weights=(0,1,0,0))
    #     # bleu_score_3=sentence_bleu(gt,pred,weights=(0,0,1,0))
    #     # bleu_score_4=sentence_bleu(gt,pred,weights=(0,0,0,1))
    #     # cider_score=self.cider_scorer.compute_score(gt,pred)
    #     # rouge_score=self.rouge(gt,pred)
    #     # score,scores=get_eval_score(gt,pred)
    #     print(scores)
    #     return scores
        #return [bleu_score_1,bleu_score_2,bleu_score_3,bleu_score_4,meteor_score,cider_score,rouge_score]


    def get_best(self,crr,pre):
        for i in range(0,len(crr)-1):
            crr[i]=max(crr[i],pre[i])
        return crr

# def save_result(result, result_dir, filename, remove_duplicate=''):
#     """Saving predictions as json file for evaluation."""

#     # combine results from all processes
#     result_new = []

#     if remove_duplicate:
#         result_new = []
#         id_list = []
#         for res in track_iter_progress(result):
#             if res[remove_duplicate] not in id_list:
#                 id_list.append(res[remove_duplicate])
#                 result_new.append(res)
#         result = result_new

#     final_result_file_url = os.path.join(result_dir, '%s.json' % filename)
#     print(f'result file saved to {final_result_file_url}')
#     json.dump(result, open(final_result_file_url, 'w'))

#     return final_result_file_url


# def coco_caption_eval(results_file, ann_file):
#     """Evaluation between gt json and prediction json files."""
#     # create coco object and coco_result object
#     # coco = mmengine.load(self.ann_file)
#     # coco_result = coco.loadRes(results_file)
# 	#coco = COCO(ann_file
#     # # create coco_eval object by taking coco and coco_result
#     # coco_eval = COCOEvalCap(coco, coco_result)

#     # # make sure the image ids are the same
#     # coco_eval.params['image_id'] = coco_result.getImgIds()

#     # # This will take some times at the first run
#     # coco_eval.evaluate()

#     # # print output evaluation scores
#     # for metric, score in coco_eval.eval.items():
#     #     print(f'{metric}: {score:.3f}')
#     #return coco_eval.eval
#     annotation=mmengine.load(ann_file)
#     result=mmengine.load(results_file)
#     # print(annotation['images'])
#     # print(result)
#     return None
  
@METRICS.register_module()
class CDMetric(BaseMetric):
	default_prefix: Optional[str] = 'cd'

	def __init__(self,
				 ignore_index: int = 255,
				 collect_device: str = 'cpu',
				 prefix: Optional[str] = None,save_path='/home/zys/wnmd/fuck/fuck/test/imgs/pred/',
				 **kwargs) -> None:
		super().__init__(collect_device=collect_device, prefix=prefix)
		self.ignore_index = ignore_index
		self.save_path = save_path
	def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
		for data_sample in data_samples:
			pred_label = data_sample['pred_sem_seg']['data'].squeeze()
			# format_only always for test dataset without ground truth
   #pad_shape', 'ori_shape', 'img_padding_size', 'seg_map_path', 'img_path', 'img_shape', 'scale_factor', 'gt_sem_seg', 'pred_sem_seg', 'seg_logits'
			gt_label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label)
			path=  data_sample['seg_map_path'].split('/')[-1]
			img=pred_label.cpu().numpy()
			img[img!=0]=255
			cv2.imwrite(self.save_path+'/'+path,img)
			self.results.append((pred_label, gt_label))

	def compute_metrics(self, results: list) -> Dict[str, float]:
		num_classes = len(self.dataset_meta['classes'])
		class_names = self.dataset_meta['classes']

		assert num_classes == 2, 'Only support binary classification in CDMetric.'

		logger: MMLogger = MMLogger.get_current_instance()
		pred_label, label = zip(*results)
		preds = torch.stack(pred_label, dim=0)
		target = torch.stack(label, dim=0)

		multiclass_precision_ = multiclass_precision(preds, target, num_classes=num_classes, average=None, ignore_index=self.ignore_index)
		multiclass_recall_ = multiclass_recall(preds, target, num_classes=num_classes, average=None, ignore_index=self.ignore_index)
		multiclass_f1_score_ = multiclass_f1_score(preds, target, num_classes=num_classes, average=None, ignore_index=self.ignore_index)
		multiclass_jaccard_index_ = multiclass_jaccard_index(preds, target, num_classes=num_classes, average=None, ignore_index=self.ignore_index)
		accuracy_ = multiclass_accuracy(preds, target, num_classes=num_classes, average=None, ignore_index=self.ignore_index)
		binary_accuracy_ = binary_accuracy(preds, target, ignore_index=self.ignore_index)
		ret_metrics = OrderedDict({
			'acc': accuracy_.cpu().numpy(),
			'p': multiclass_precision_.cpu().numpy(),
			'r': multiclass_recall_.cpu().numpy(),
			'f1': multiclass_f1_score_.cpu().numpy(),
			'iou': multiclass_jaccard_index_.cpu().numpy(),
			'macc': binary_accuracy_.cpu().numpy(),
		})

		metrics = dict()
		for k, v in ret_metrics.items():
			if k == 'macc':
				metrics[k] = v.item()
			else:
				for i in range(num_classes):
					metrics[k + '_' + class_names[i]] = v[i].item()

		# each class table
		ret_metrics.pop('macc', None)
		ret_metrics_class = OrderedDict({
			ret_metric: np.round(ret_metric_value * 100, 2)
			for ret_metric, ret_metric_value in ret_metrics.items()
		})
		ret_metrics_class.update({'Class': class_names})
		ret_metrics_class.move_to_end('Class', last=False)
		class_table_data = PrettyTable()
		for key, val in ret_metrics_class.items():
			class_table_data.add_column(key, val)

		print_log('per class results:', logger)
		print_log('\n' + class_table_data.get_string(), logger=logger)
		return metrics