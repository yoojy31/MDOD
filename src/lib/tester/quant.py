import os
import sys
import tqdm
import pickle
import numpy as np
from lib import util as lib_util
from lib.tester import __util__ as test_util
from lib.external.dataset import factory
from .__abc__ import TesterABC


class QuantTester(TesterABC):
    def __init__(self, global_args, tester_args):
        super(QuantTester, self).__init__(global_args, tester_args)
        self.n_classes = global_args['n_classes']
        self.imdb_name = tester_args['dataset']
        assert self.imdb_name in ('voc_2007_test', 'coco_2017_val', 'coco_2017_test-dev')

    def run(self, framework, data_loader, result_dir):
        assert data_loader.batch_size == 1
        num_samples = data_loader.dataset.__len__()
        all_boxes = [[[] for _ in range(num_samples)] for _ in range(self.n_classes)]
        # temp_boxes = []

        nms_ratios = list()
        infer_times, net_times, pp_times = list(), list(), list()
        sample_pbar = tqdm.tqdm(data_loader)
        for i, data_dict in enumerate(sample_pbar):

            output_dict, result_dict, value_dict = framework.infer_forward(data_dict)
            if i > 5:
                infer_times.append(lib_util.cvt_torch2numpy(value_dict['infer_time']))
                net_times.append(lib_util.cvt_torch2numpy(value_dict['net_time']))
                pp_times.append(lib_util.cvt_torch2numpy(value_dict['pp_time']))

                if ('nms_ratio' in value_dict.keys()) and (value_dict['nms_ratio'] is not None):
                    nms_ratios.append(lib_util.cvt_torch2numpy(value_dict['nms_ratio']))
                    sample_pbar.set_description(
                        'infer: %.3, nms_ratio: %.3f' % (np.mean(infer_times), np.mean(nms_ratios)))
                else:
                    sample_pbar.set_description('infer: %.3f, net: %.3f, pp: %.3f' % (
                        np.mean(infer_times), np.mean(net_times), np.mean(pp_times)))

            # total predict boxes shape : (batch, # pred box, 4)
            # total predict boxes confidence shape : (batch, # pred box, 1)
            # total predict boxes label shape : (batch, # pred box, 1)
            img_size_s = data_dict['img_size'].float()[0]
            input_size = data_dict['img'].shape[2:]

            boxes_s = result_dict['boxes_l'][0]
            confs_s = result_dict['confs_l'][0]
            labels_s = result_dict['labels_l'][0]

            boxes_s = lib_util.cvt_torch2numpy(boxes_s)
            confs_s = lib_util.cvt_torch2numpy(confs_s)
            labels_s = lib_util.cvt_torch2numpy(labels_s)
            boxes_s, confs_s, labels_s = \
                lib_util.sort_boxes_s_np(boxes_s, confs_s, labels_s)

            boxes_s[:, [0, 2]] *= (float(img_size_s[1]) / float(input_size[1]))
            boxes_s[:, [1, 3]] *= (float(img_size_s[0]) / float(input_size[0]))

            if len(confs_s.shape) == 1:
                confs_s = np.expand_dims(confs_s, axis=1)

            for cls_box, cls_conf, cls_label in zip(boxes_s, confs_s, labels_s):
                cls_box_with_conf = np.concatenate((cls_box, cls_conf), axis=0)
                cls_box_with_conf = np.expand_dims(cls_box_with_conf, axis=0)
                all_boxes[int(cls_label)][i].append(cls_box_with_conf)

            for c in range(self.n_classes):
                all_boxes[c][i] = np.concatenate(all_boxes[c][i], axis=0) \
                    if len(all_boxes[c][i]) > 0 else np.concatenate([[]], axis=0)

            data_dict.clear()
            del data_dict

        # create result directories
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        dataset_root = data_loader.dataset.get_dataset_roots()[0]
        imdb = factory.get_imdb(self.imdb_name, dataset_root)

        if 'coco' in self.imdb_name:
            sys_stdout = sys.stdout
            result_file_path = open(os.path.join(result_dir, 'ap_ar.txt'), 'w')
            sys.stdout = result_file_path
            imdb.evaluate_detections(all_boxes, result_dir)
            print('inference time: %.5f sec' % (np.mean(infer_times)))
            print('network time: %.5f sec' % (np.mean(net_times)))
            print('post-proc time: %.5f sec' % (np.mean(pp_times)))
            sys.stdout = sys_stdout
            result_file_path.close()

        else:
            det_file_path = os.path.join(result_dir, 'detection_results.pkl')
            with open(det_file_path, 'wb') as det_file:
                pickle.dump(all_boxes, det_file, pickle.HIGHEST_PROTOCOL)
            result_dict = imdb.evaluate_detections(all_boxes, result_dir)
            result_file_path = os.path.join(result_dir, 'mean_ap.txt')

            with open(result_file_path, 'w') as file:
                file.write(result_dict['msg'])
                file.write('\ninfer_time: %.4f\nnms_ratio: %.4f' %
                           (np.mean(infer_times), np.mean(nms_ratios)))

            os.mkdir(os.path.join(result_dir, 'plot'))
            for key in result_dict.keys():
                if key is not 'msg':
                    sns_figure = test_util.draw_rec_prec_graph(
                        result_dict[key]['rec'], result_dict[key]['prec'])
                    result_file_path = os.path.join(result_dir, 'plot/%s.png' % key)
                    sns_figure.savefig(result_file_path)

            os.remove(det_file_path)
        all_boxes.clear()


