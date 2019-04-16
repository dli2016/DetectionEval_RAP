import numpy as np
from progress_bar import *
from operator import itemgetter

def calIOU(rect1, rect2):
    # Rect1
    x1 = rect1['x']
    y1 = rect1['y']
    width1 = rect1['width']
    height1= rect1['height']

    # Rect2
    x2 = rect2['x']
    y2 = rect2['y']
    width2 = rect2['width']
    height2 = rect2['height']

    # Calculate bound
    end_x = max(x1+width1, x2+width2)
    start_x = min(x1, x2)
    width = width1 + width2 - (end_x-start_x)
    end_y = max(y1+height1, y2+height2)
    start_y = min(y1, y2)
    height= height1 + height2 - (end_y-start_y)

    # Calculate area and ratio
    if width <= 0 or height <= 0:
        ratio = 0
    else:
        area = width * height
        area1= width1 * height1
        area2= width2 * height2
        ratio = area * 1.0 / (area1 + area2 - area)
    return ratio

def calAP(detections, gt_pos, gt_neg, threshold):
    # Type changing.
    for d in detections:
        d['score'] = float(d['score'])
    # Sort
    d_sorted = sorted(detections, key=itemgetter('score'), reverse=True)
    # calculate ap
    npos = len(gt_pos)
    nd = len(d_sorted)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    # Progress bar
    bar = ProgressBar(total = nd)

    # Calculate
    cnt = 0
    for d in d_sorted:
        d_fn = d['filename'].split('.')[0] + '.png'
        checked_pos = filter(lambda item: item['filename']==d_fn, gt_pos)
        checked_neg = filter(lambda item: item['filename']==d_fn, gt_neg)
        d_bb = {'x': int(float(d['x'])), 'y': int(float(d['y'])), \
                'width': int(float(d['width'])), \
                'height': int(float(d['height']))}
        # Positives
        current_cnt = 0
        if len(checked_pos) > 0:
            for pos in checked_pos:
                pos_bb = {'x': int(pos['x']), 'y': int(pos['y']),\
                          'width': int(pos['width']), \
                          'height': int(pos['height'])}
                iou = calIOU(d_bb, pos_bb)
                if iou > threshold:
                    current_cnt = current_cnt + 1
            tp[cnt] = current_cnt
        # Negatives
        current_cnt = 0
        if len(checked_neg) > 0:
            for neg in checked_neg:
                neg_bb = {'x': int(neg['x']), 'y': int(neg['y']),\
                          'width': int(neg['width']), \
                          'height': int(neg['height'])}
                iou = calIOU(d_bb, neg_bb)
                if iou > threshold:
                    current_cnt = current_cnt + 1
            fp[cnt] = current_cnt
        cnt = cnt + 1
        bar.move()
        bar.draw()
        #print "%d/%d" % (cnt, nd)
    # Precision and recall
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    rec = tp / float(npos)
    prec= tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # Calculate AP
    ap = vocAP(rec, prec)

    return tp[-1], fp[-1], ap

def vocAP(rec, prec, use_07_metric=True):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
