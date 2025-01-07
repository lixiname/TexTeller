import os
import cv2 as cv
from pathlib import Path
import onnxruntime
from onnxruntime import InferenceSession
from models.thrid_party.paddleocr.infer import predict_det, predict_rec
from models.thrid_party.paddleocr.infer import utility
from models.ocr_model.model.TexTeller import TexTeller
from models.det_model.inference import PredictConfig
from src.enumUtils import RuntimeTypeEnum, AlgorithmsEnum
from src.models.utils.mix_ali_inference import ali_inference
from src.recognition_onnx import load_recognition_onnx
if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    print(onnxruntime.__version__)
    print(f'onnxruntime version:{onnxruntime.__version__}')
    print(os.getcwd())
    inference_mode = 'cuda'
    print('Loading model and tokenizer...')
    latex_rec_model = TexTeller.from_pretrained()
    tokenizer = TexTeller.get_tokenizer()
    print('Model and tokenizer loaded.')
    img_path = str(Path('../ori/1/2.jpg'))
    img = cv.imread(img_path)
    print('Inference...')
    infer_config = PredictConfig("./models/det_model/model/infer_cfg.yml")
    latex_det_model = InferenceSession("./models/det_model/model/rtdetr_r50vd_6x_coco.onnx")

    use_gpu = True
    SIZE_LIMIT = 20 * 1024 * 1024

    text_det_tpye = AlgorithmsEnum.V1
    paddleocr_args = utility.parse_args()
    paddleocr_args.use_onnx = True
    if text_det_tpye == AlgorithmsEnum.V1:
        # ch_PP-OCRv4_server_det.onnx   default_model.onnx
        det_model_dir = "./models/thrid_party/paddleocr/checkpoints/det/ch_PP-OCRv4_server_det.onnx"

        paddleocr_args.det_model_dir = det_model_dir
        det_use_gpu = False
        # The CPU inference of the detection model will be faster than the GPU inference (in onnxruntime)
        paddleocr_args.use_gpu = det_use_gpu
        detector = predict_det.TextDetector(paddleocr_args)

    elif text_det_tpye == AlgorithmsEnum.V2:
        det_model_dir = Path("./models/thrid_party/det")
        detector = onnxruntime.InferenceSession(det_model_dir / 'cbn_resnet18_8.onnx', providers=[RuntimeTypeEnum.CPU.value])

    rec_ali = False
    if rec_ali:
        rec_model_dir = Path("./models/thrid_party/ali/rec")
        runtimeType = RuntimeTypeEnum.GPU
        recognizer = load_recognition_onnx(rec_model_dir, runtimeType=runtimeType)
    else:
        # ch_PP-OCRv4_server_rec.onnx  default_model.onnx
        rec_model_dir = "./models/thrid_party/paddleocr/checkpoints/rec/ch_PP-OCRv4_server_rec.onnx"
        rec_use_gpu = use_gpu and not (os.path.getsize(rec_model_dir) < SIZE_LIMIT)
        paddleocr_args.rec_model_dir = rec_model_dir
        paddleocr_args.use_gpu = rec_use_gpu
        recognizer = predict_rec.TextRecognizer(paddleocr_args)

    lang_ocr_models = [detector, recognizer]
    latex_rec_models = [latex_rec_model, tokenizer]
    res = ali_inference(img_path, infer_config, latex_det_model, lang_ocr_models, latex_rec_models,
                        inference_mode, 1, rec_ali, text_det_tpye)
    print(res)
