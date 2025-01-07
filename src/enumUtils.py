from enum import Enum

class AlgorithmsEnum(Enum):
    V1 = 1  # 默认
    V2 = 2  # cbn深度学习算法

class LanguageEnum(Enum):
    US = 1  # 英语
    CN = 2  # 简体中文

class RuntimeTypeEnum(Enum):
    GPU = 'CUDAExecutionProvider'
    CPU = 'CPUExecutionProvider'
    GPUsuper = 'Tensorrt'
    TensorCore = 'TensorrtExecutionProvider'


LOG_SUCCESS = 'SUCCESS:'
LOG_ERROR = 'ERROR:'
LOG_INFO = 'INFO:'
