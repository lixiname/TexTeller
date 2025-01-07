from pathlib import Path
import numpy as np
import onnxruntime as rt
import cv2

from enumUtils import RuntimeTypeEnum


def keepratio_resize(img):
    """
    :param img: 重新缩放输入img到指定比例
    :return: img
    """
    #img = img[:, :int(img.shape[1]/2), :]
    try:
        cur_ratio = img.shape[1] / float(img.shape[0])
    except:
        print(img.shape)
        raise

    mask_height = 32
    mask_width = 804
    require_ratio = float(mask_width) / mask_height
    if cur_ratio > require_ratio:
        cur_target_height = mask_height
        cur_target_width = mask_width
    else:
        cur_target_height = mask_height
        cur_target_width = int(mask_height * cur_ratio)

    img = cv2.resize(img, (cur_target_width, cur_target_height))
    mask = np.zeros([mask_height, mask_width, 3]).astype(np.uint8)
    mask[:img.shape[0], :img.shape[1], :] = img
    img = mask
    return img



def map_to_char(res, labelMapping:dict):
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # 减去最大值以提高数值稳定性
        return e_x / e_x.sum(axis=-1, keepdims=True)

    outprobs = softmax(res[0])
    preds = np.argmax(outprobs, axis=-1)
    batchSize, length = preds.shape
    final_str_list = []
    char_list = []
    for i in range(batchSize):
        pred_idx = preds[i].tolist()
        last_p = 0
        str_pred = []
        for index, p in enumerate(pred_idx):
            if p != last_p and p != 0:
                str_pred.append(labelMapping[p])
                char_list.append({
                    'char': labelMapping[p],
                    'probability': round(float(outprobs[0][index][p]), 3)
                })
            last_p = p
        final_str = ''.join(str_pred)
        final_str_list.append(final_str)
    res_str = ''.join(final_str_list)
    return res_str, char_list

def load_vocab(model_path: Path):
    labelMapping = dict()
    with open(model_path / 'vocab.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        cnt = 2
        for line in lines:
            line = line.strip('\n')
            labelMapping[cnt] = line
            cnt += 1
    return labelMapping


def resize_rec_input_size(crop_img):
    img = keepratio_resize(crop_img)
    # img_debug = img.copy()
    img = img.astype(np.float32)
    chunk_img = []
    for i in range(3):
        left = (300 - 48) * i
        chunk_img.append(img[:, left:left + 300, :])
    merge_img = np.concatenate(chunk_img, axis=0)
    # 3 3 32 300
    data = np.reshape(merge_img, (3, 32, 300, 3)) / 255.
    data = np.transpose(data, (0, 3, 1, 2))
    input_data = data
    return input_data

def load_recognition_onnx(model_path: Path, runtimeType: RuntimeTypeEnum):
    if runtimeType == RuntimeTypeEnum.CPU:
        sess = rt.InferenceSession(model_path / 'modelscope_recognition_handwritten.onnx',
                                   providers=[runtimeType.value])
    elif runtimeType == RuntimeTypeEnum.GPU:
        sess = rt.InferenceSession(model_path / 'modelscope_recognition_handwritten.onnx',
                                   providers=[runtimeType.value, RuntimeTypeEnum.CPU.value])
    else:
        sess = rt.InferenceSession(model_path / 'modelscope_recognition_handwritten.onnx',
                                   providers=[runtimeType.value, RuntimeTypeEnum.GPU.value])

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    labelMapping = load_vocab(model_path)
    def recognition(crop_img):
        input_data = resize_rec_input_size(crop_img)
        res = sess.run([output_name], {input_name: input_data})
        # 彩图的rgb通道一直没管，即按照的cv2的bgr顺序输入的model
        res_str, char_list = map_to_char(res, labelMapping)
        return res_str, char_list


    return recognition



def crop_img_by_detection(full_img, points_list):
    pairs = [(points_list[i], points_list[i + 1]) for i in range(0, len(points_list), 2)]
    x_start = pairs[0][0]
    y_start = pairs[0][1]
    x_end = pairs[2][0]
    y_end = pairs[2][1]
    crop_img = full_img[y_start:y_end+1, x_start:x_end+1, :]
    #cv2.imshow('', crop_img)
    #cv2.waitKey(0)
    return crop_img
