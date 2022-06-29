import numpy as np
import cv2
import pandas as pd

pth = './output/car.jpg'

rois = []
areas = []
damage = []
posi_list = []
damage_list = []


def start(img):
    classes = ['car']
    net = cv2.dnn.readNetFromDarknet('models/yolov4_car_or_nocar.cfg', 'models/yolov4_car_or_nocar.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    ##############################
    nmsThreshold = 0.2
    bbox = []
    classIds = []
    confs = []
    ht, wt, ct = img.shape
    try:
        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                obj = classes[classId]
                conf = scores[classId]
                if (obj == 'car') and (conf > 0.4):
                    w, h = int(det[2] * wt), int(det[3] * ht)
                    x, y = int((det[0] * wt) - w / 2), int((det[1] * ht) - h / 2)
                    bbox.append([x, y, w, h])
                    classIds.append(classId)
                    confs.append(float(conf))

        # scaning car image
        indices = cv2.dnn.NMSBoxes(bbox, confs, 0.3, nmsThreshold)
        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            roi = img[y:y + h, x:x + w]
            areas.append(w * h)
            rois.append(roi)

    except:
        pass
    ##############################################
    if len(rois) == 0:
        txt = 'Try with another image..'
        txt2 = 'Put car image..'
        img = cv2.resize(img,(400,300))
        img = cv2.rectangle(img, (40, 200), (360, 270), (140, 0, 0), cv2.FILLED)
        img = cv2.putText(img, txt2, (50, 244), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 200, 200), 3)
        cv2.imwrite('./output/object.jpg', img)

    elif (len(rois) > 0):
        txt = 'car'
        indx = areas.index(np.max(areas))
        roi = rois[indx]
        roi = cv2.resize(roi, (400, 300))
        cv2.rectangle(roi, (0, 0), (40, 24), (255, 0, 50), cv2.FILLED)
        cv2.putText(roi, txt, (0, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite('output/car.jpg', roi)
    return txt


##########################################################
# detecting Damage or No-damage
##########################################################


def find_damage(outs, imag):
    clas = ['damage', 'noDamage']
    nmsThres = 0.3
    ht, wt, ct = imag.shape
    bbox = []
    classIds = []
    confs = []
    try:
        for output in outs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                obj = clas[classId]
                conf = scores[classId]
                w, h = int(det[2] * wt), int(det[3] * ht)
                x, y = int((det[0] * wt) - w / 2), int((det[1] * ht) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(conf))

        # scaning car image
        indices = cv2.dnn.NMSBoxes(bbox, confs, 0.4, nmsThres)
        for i in indices:
            i = i[0]
            box = bbox[i]
            txt = clas[classIds[i]]
            damage.append(txt)
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(imag, (214, 0), (400, 24), (255, 50, 0), cv2.FILLED)
            cv2.putText(imag, f'found {txt}', (216, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    except:
        pass


def if_damage(rois):
    if len(rois) > 0:
        nete = cv2.dnn.readNetFromDarknet('models/yolov4_damage_nodamage.cfg', 'models/yolov4_damage_nodamage.weights')
        nete.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        nete.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        imag = cv2.imread(pth)
        imag = cv2.resize(imag, (400, 300))
        blobe = cv2.dnn.blobFromImage(imag, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)
        nete.setInput(blobe)
        layerName = nete.getLayerNames()
        outputNums = [layerName[i[0] - 1] for i in nete.getUnconnectedOutLayers()]
        outs = nete.forward(outputNums)
        find_damage(outs, imag)
        cv2.imwrite('output/damage.jpg', imag)
    else:
        pass


#######################################################################
# Detecting position i,e side, rear or front
######################################################################

def findPosition(outs, imag):
    clas = ['Front', 'Side', 'Rear']
    nmsThres = 0.3
    ht, wt, ct = imag.shape
    bbox = []
    classIds = []
    confs = []
    try:
        for output in outs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                obj = clas[classId]
                conf = scores[classId]
                w, h = int(det[2] * wt), int(det[3] * ht)
                x, y = int((det[0] * wt) - w / 2), int((det[1] * ht) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(conf))

        # scaning car image
        indices = cv2.dnn.NMSBoxes(bbox, confs, 0.4, nmsThres)
        for i in indices:
            i = i[0]
            box = bbox[i]
            txt = clas[classIds[i]]
            posi_list.append(txt)
            x, y, w, h = box[0], box[1], box[2], box[3]
            imag = cv2.rectangle(imag, (x + int(w / 2) - 26, y + int(h / 2) - 10),
                                  (x + 54 + int(w / 2), y + 28 + int(h / 2)), (255, 0, 255), cv2.FILLED)
            cv2.putText(imag, txt, (x - 20 + int(w / 2), y + 16 + int(h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 3)
    except:
        pass


def position(rois):
    if len(rois) > 0:
        nete = cv2.dnn.readNetFromDarknet('models/yolov4_location(side,rear,front).cfg', 'models/yolov4_location(side,rear,front).weights')
        nete.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        nete.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        imag = cv2.imread(pth)
        imag = cv2.resize(imag, (400, 300))
        blobe = cv2.dnn.blobFromImage(imag, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)
        nete.setInput(blobe)
        layerName = nete.getLayerNames()
        outputNums = [layerName[i[0] - 1] for i in nete.getUnconnectedOutLayers()]
        outs = nete.forward(outputNums)
        findPosition(outs, imag)
        cv2.imwrite('output/position.jpg', imag)
    else:
        pass


###################################################################
# Detecting Damage quality
##################################################################

def findDamage(outs, imag):
    clas = ['severe', 'moderate', 'minor']
    nmsThres = 0.3
    ht, wt, ct = imag.shape
    bbox = []
    classIds = []
    confs = []

    try:
        for output in outs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                obj = clas[classId]
                conf = scores[classId]
                w, h = int(det[2] * wt), int(det[3] * ht)
                x, y = int((det[0] * wt) - w / 2), int((det[1] * ht) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(conf))

        # scaning car image
        indices = cv2.dnn.NMSBoxes(bbox, confs, 0.4, nmsThres)
        for i in indices:
            i = i[0]
            box = bbox[i]
            txt = clas[classIds[i]]
            damage_list.append(txt)
            x, y, w, h = box[0], box[1], box[2], box[3]
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            imag = cv2.rectangle(imag, pt1, pt2, (255, 100, 255), cv2.FILLED)
            imag = cv2.rectangle(imag, (x, y - 14), (x + 105, y), (255, 0, 0), cv2.FILLED)
            cv2.putText(imag, txt, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    except:
        pass


def Damage():
    nete = cv2.dnn.readNetFromDarknet('models/yolov4_damage_condition(minor,moderate,severe).cfg', 'models/yolov4_damage_condition(minor,moderate,severe).weights')
    nete.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    nete.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    imag = cv2.imread(pth)
    imag = cv2.resize(imag, (400, 300))
    blobe = cv2.dnn.blobFromImage(imag, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)
    nete.setInput(blobe)
    layerName = nete.getLayerNames()
    outputNums = [layerName[i[0] - 1] for i in nete.getUnconnectedOutLayers()]
    outs = nete.forward(outputNums)
    findDamage(outs, imag)
    cv2.imwrite('output/damage_cond.jpg', imag)


####################################################################
# processing data and making results
###################################################################

con = []
pos = []
amount = []


def results():
    if 'severe' in damage_list:
        con.append('Severe')
    elif ('severe' not in damage_list) and ('moderate' in damage_list):
        con.append('Moderate')
    elif ('severe' not in damage_list) and ('moderate' not in damage_list) and ('minor' in damage_list):
        con.append('Minor')

    if 'Front' in posi_list:
        pos.append('Front')
    elif 'Rear' in posi_list:
        pos.append('Rear')
    elif ('Front' not in posi_list) and ('Rear' not in posi_list) and ('Side' in posi_list):
        pos.append('Side')

    if ('Front' in pos) and ('Severe' in con):
        am = '$510-1200'
        amount.append(am)
    elif ('Side' in pos) and ('Severe' in con):
        am = '$410-1000'
        amount.append(am)
    elif ('Rear' in pos) and ('Severe' in con):
        am = '$510-1200'
        amount.append(am)

    if ('Front' in pos) and ('Moderate' in con):
        am = '$200-500'
        amount.append(am)
    elif ('Side' in pos) and ('Moderate' in con):
        am = '$200-400'
        amount.append(am)
    elif ('Rear' in pos) and ('Moderate' in con):
        am = '$200-500'
        amount.append(am)

    if ('Front' in pos) and ('Minor' in con):
        am = '$75-200'
        amount.append(am)
    elif ('Side' in pos) and ('Minor' in con):
        am = '$50-200'
        amount.append(am)
    elif ('Rear' in pos) and ('Minor' in con):
        am = '$75-200'
        amount.append(am)


#########################################################
# recording final results
#########################################################

def make_results(df):
    car_e = list(df['car_exist']) + [len(rois)]
    cond = list(df['condition']) + [damage[0]]
    posi = list(df['position']) + pos
    damage_c = list(df['damage_condition']) + con
    est_a = list(df['Estimated_Amount']) + amount
    dict = {'car_exist': car_e, 'condition': cond, 'position': posi, 'damage_condition': damage_c,
            'Estimated_Amount': est_a}
    Df = pd.DataFrame(dict)
    Df.to_csv('result.csv')


#######################################################
# deleting records if necessery
#######################################################

def del_results():
    dict = {'car_exist': [], 'condition': [], 'position': [], 'damage_condition': [], 'Estimated_Amount': []}
    df = pd.DataFrame(dict)
    df.to_csv('result.csv')

