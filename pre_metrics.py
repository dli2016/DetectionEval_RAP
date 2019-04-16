import os
import sys
import time
import numpy as np
from util.file_operation import *
from util.calculation import *

class Metric:
    def __init__(self, pos_ground_truth_filename, neg_ground_truth_filename):
        self.__pos_gt = loadCSV(pos_ground_truth_filename)
        self.__neg_gt = loadCSV(neg_ground_truth_filename)

    def getPosNum(self):
        return len(self.__pos_gt)

    def calPR(self, res_filename, iou, confidence):
        # Filename
        res_filename_splitted = res_filename.split('/')
        res_filename_splitted[-1] = res_filename_splitted[-1].split('.')[0]
        save_temp_path = ''
        for splitted_str in res_filename_splitted:
            save_temp_path = save_temp_path + splitted_str + '-'
        save_temp_path = save_temp_path + 'iou' + str(iou) + '-conf' + \
            str(confidence)
        save_path_tp = save_temp_path + '-tp.csv'
        save_path_fp = save_temp_path + '-fp.csv'
        # Check
        file_format = res_filename.split('.')[-1]
        if file_format == 'csv':
            detected_res = loadCSV(res_filename)
            detected_res_refined = self.__refineCSV(detected_res, confidence)
            pos_gt = self.__pos_gt
            neg_gt = self.__neg_gt
            # TP
            print '======== TP ========'
            TPs = self.__checkCSV(pos_gt, detected_res_refined, iou)
            TP_num = len(TPs)
            print '#TP: %d' % TP_num
            # FP
            print '======== FP ========'
            FPs = self.__checkCSV(neg_gt, detected_res_refined, iou)
            FP_num = len(FPs)
            print '#FP: %d' % FP_num
            saveCSV(save_path_tp, TPs)
            saveCSV(save_path_fp, FPs)
        elif file_format == 'json':
            detected_res = loadJson(res_filename)
        else:
            detected_res = None

    def getAP(self, det_filename, iou):
        # Check file type
        file_type = det_filename.split('.')[-1]
        if file_type == 'csv':
            detected_res = loadCSV(det_filename)
        elif file_type == 'json':
            is_csv_existed = os.path.exists(det_filename.split('.')[0] + '.csv')
            if not is_csv_existed:
                detected_res_temp = loadJson(det_filename)
                detected_res = []
                print 'Format changing ...'
                keys = detected_res_temp.keys()
                cnt = 0
                for key in keys:
                    dets = detected_res_temp[key]
                    for det in dets:
                        score = det['soc']
                        bb = det['loc']
                        item = {'index': cnt,\
                                'filename': key, \
                                'score': score,\
                                'x': bb[0], \
                                'y': bb[1], \
                                'width': int(bb[2])-int(bb[0]),\
                                'height': int(bb[3])-int(bb[1])}
                        detected_res.append(item)
                        cnt = cnt + 1
                saveCSV(det_filename.split('.')[0]+'.csv', detected_res)
                print 'Changing done!'
            else:
                detected_res = loadCSV(det_filename)
        # Get ap
        pos_gt = self.__pos_gt
        neg_gt = self.__neg_gt
        ntp, nfp, ap = calAP(detected_res, pos_gt, neg_gt, iou)
        return ntp, nfp, ap

    def __checkCSV(self, probes, galleries, iou):
        checked_res = []
        checked_num = 0
        total_pb_num= len(probes)
        cnt = 0
        for pb in probes:
            pb_fn = pb['filename'].split('.')[0] + '.jpg'
            checked_gly = filter(lambda item: item['filename']==pb_fn,\
                galleries)
            rect_pb = {'x': int(pb['x']), 'y': int(pb['y']), 'width':\
                int(pb['width']), 'height': int(pb['height'])}
            if len(checked_gly) > 0:
                for gly in checked_gly:
                    rect_gly = {'x': int(float(gly['x'])), \
                                'y': int(float(gly['y'])), \
                                'width': int(float(gly['width'])), \
                                'height': int(float(gly['height']))}
                    iou_cal = calIOU(rect_pb, rect_gly)
                    if iou_cal > iou:
                        gly['iou'] = iou_cal
                        checked_res.append(gly)
            cnt = cnt + 1
            print '%d/%d' % (cnt, total_pb_num)
        return checked_res

    def __refineCSV(self, detected_res, conf):
        res = filter(lambda item: float(item['score'])>conf, detected_res)
        return res

    def __refineJson(self):
        return res

def run(detected_res_file_path, iou, conf):
    pos_ground_truth_filename = 'positive_boundingboxes.csv'
    neg_ground_truth_filename = 'negative_boundingboxes_refined.csv'

    metric = Metric(pos_ground_truth_filename, neg_ground_truth_filename)
    #metric.calPR(detected_res_file_path, iou, conf)
    ntp, nfp, ap = metric.getAP(detected_res_file_path, iou)
    print 'AP: %f, #TPs: %d, #FPs: %d' % (ap, ntp, nfp)

def run_v2(root_dir, iou):
    sub_dirs = os.listdir(root_dir)
    filenames = []
    for sub_dir in sub_dirs:
        sub_dir = os.path.join(root_dir, sub_dir)
        if os.path.isdir(sub_dir):
            filenames_temp = collectAllFiles(sub_dir)
            filenames = filenames + filenames_temp
        else:
            continue
    # calculate
    pos_ground_truth_filename = 'positive_boundingboxes.csv'
    neg_ground_truth_filename = 'negative_boundingboxes_refined.csv'
    metric = Metric(pos_ground_truth_filename, neg_ground_truth_filename)
    res = []
    np = metric.getPosNum()
    for filename in filenames:
        print '======================================================'
        print filename + ':'
        ntp, nfp, ap = metric.getAP(filename, iou)
        prec = float(ntp) / (ntp + nfp)
        rec = float(ntp) / np
        f1 = 2.0*prec*rec / (prec+rec)
        name = filename.split('.')[0]
        gp = name.split('/')[1]
        method = name.split('/')[-1]
        item = {'group': gp, 'method': method, 'tp': ntp, 'fp': nfp, \
            'prec': prec, 'rec': rec, 'f1': f1, 'AP': ap}
        res.append(item)
        print item
    saveCSV('eval_results.csv', res);

if __name__=='__main__':
    filename = sys.argv[1]
    iou = float(sys.argv[2])
    conf= float(sys.argv[3])
    run(filename, iou, conf)
    #dir_rt = sys.argv[1]
    #iou = float(sys.argv[2])
    #run_v2(dir_rt, iou)
    #run(dir_rt, iou, 0.5)
