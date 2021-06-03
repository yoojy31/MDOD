import torch
import torch.nn.functional as func
from .__abc__ import LossFunctionABC
from . import __func__ as loss_func
from lib import util


class MDODLossFunc(LossFunctionABC):
    def __init__(self, global_args, loss_func_args):
        super(MDODLossFunc, self).__init__(global_args, loss_func_args)
        self.coord_pdf = util.cauchy_pdf \
            if loss_func_args['coord_pdf'] == 'cauchy' \
            else util.gaussian_pdf

        self.mog_pi_thresh = loss_func_args['mog_pi_thresh']
        self.mod_pi_thresh = loss_func_args['mod_pi_thresh']
        self.mod_n_samples = loss_func_args['mod_n_samples']
        self.mod_max_samples = loss_func_args['mod_max_samples']

        self.sampling_noise = loss_func_args['sampling_noise']
        self.value_return = loss_func_args['value_return']
        self.n_classes = global_args['n_classes']

        assert 'mog_nll' in self.lw_dict.keys()
        assert 'mod_nll' in self.lw_dict.keys() 
        assert loss_func_args['coord_pdf'] in ('gaussian', 'cauchy')

    def forward(self, output_dict, data_dict):
        mu, sig = output_dict['mu'], output_dict['sig']
        prob, pi = output_dict['prob'], output_dict['pi']
        boxes, labels = data_dict['boxes'], data_dict['labels']
        n_boxes = data_dict['n_boxes']

        zero_box = False
        if torch.sum(n_boxes) == 0:
            n_boxes[0] += 1
            zero_box = True

        loss_dict = {}
        value_dict = {}
        if self.lw_dict['mog_nll'] > 0:
            max_nll = loss_func.calc_mog_nll(
                mu, sig, pi, boxes, n_boxes, self.coord_pdf, self.mog_pi_thresh, self.value_return)
            max_nll = max_nll[~torch.isnan(max_nll)]
            loss_dict.update({'mog_nll': self.lw_dict['mog_nll'] * max_nll})

        if self.lw_dict['mod_nll'] > 0:
            mod_nll_loss_return = loss_func.calc_mod_mm_nll(
                mu.detach(), sig.detach(), pi.detach(), prob, boxes, labels, n_boxes,
                self.mod_n_samples, self.mod_max_samples, self.mod_pi_thresh,
                self.n_classes, self.coord_pdf, self.sampling_noise)
            mod_nll_loss_return = mod_nll_loss_return[~torch.isnan(mod_nll_loss_return)]
            loss_dict.update({'mod_nll': self.lw_dict['mod_nll'] * mod_nll_loss_return})

        if zero_box:
            loss_dict['mog_nll'][0] *= 1e-30

        return loss_dict, value_dict
