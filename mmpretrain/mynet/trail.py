from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


class Scorer():
    def __init__(self, ref=None, gt=None):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            #(Spice(), "SPICE"),
        ]
    
    def compute_scores(self,ref,gt):
        total_scores = {}
        self.ref = ref
        self.gt = gt
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.3f" % (m, sc))
                total_scores["Bleu"] = score
            else:
                print("%s: %0.3f" % (method, score))
                total_scores[method] = score
        
        print('*****DONE*****')
        for key, value in total_scores.items():
            print('{}:{}'.format(key, value))
        return total_scores

if __name__ == '__main__':
    ref = {
        '1': ['go down the stairs and stop at the bottom .'],
        #'2': ['this is a cat.']
    }
    gt = {
        '1': ['Walk down the steps and stop at the bottom. ', 'Go down the stairs and wait at the bottom.',
              'Once at the top of the stairway, walk down the spiral staircase all the way to the bottom floor. Once you have left the stairs you are in a foyer and that indicates you are at your destination.'],
        #'2': ['It is a cat.', 'There is a cat over there.', 'cat over there.']
    }
    scorer = Scorer()
    ss=scorer.compute_scores(ref,gt)
    print(ss)