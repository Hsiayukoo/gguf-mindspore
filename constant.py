from enum import Enum
import abc
from typing import Any, List, TypeVar
import numpy as np
from gguf import GGMLQuantizationType


class GGUFException(Exception):
    pass


GGML_TENSOR_QUANTIZE_DICT = {
    "F32": GGMLQuantizationType.F32,
    "F16": GGMLQuantizationType.F16,
    "Q4_0": GGMLQuantizationType.Q4_0,
    "Q4_1": GGMLQuantizationType.Q4_1,
    "Q5_0": GGMLQuantizationType.Q5_0,
    "Q5_1": GGMLQuantizationType.Q5_1,
    "Q8_0": GGMLQuantizationType.Q8_0,
    "Q8_1": GGMLQuantizationType.Q8_1,
    "Q2_K": GGMLQuantizationType.Q2_K,
    "Q3_K": GGMLQuantizationType.Q3_K,
    "Q4_K": GGMLQuantizationType.Q4_K,
    "Q5_K": GGMLQuantizationType.Q5_K,
    "Q6_K": GGMLQuantizationType.Q6_K,
    "Q8_K": GGMLQuantizationType.Q8_K
}


class GGMLType(Enum):
    F32 = np.uint32(0)
    F16 = np.uint32(1)
    Q4_0 = np.uint32(2)
    Q4_1 = np.uint32(3)
    Q5_0 = np.uint32(6)
    Q5_1 = np.uint32(7)
    Q8_0 = np.uint32(8)
    Q8_1 = np.uint32(9)
    Q2_K = np.uint32(10)
    Q3_K = np.uint32(11)
    Q4_K = np.uint32(12)
    Q5_K = np.uint32(13)
    Q6_K = np.uint32(14)
    Q8_K = np.uint32(15)
    IQ2_XXS = np.uint32(16)
    IQ2_XS = np.uint32(17)
    IQ3_XXS = np.uint32(18)
    IQ1_S = np.uint32(19)
    IQ4_NL = np.uint32(20)
    IQ3_S = np.uint32(21)
    IQ2_S = np.uint32(22)
    IQ4_XS = np.uint32(23)
    I8 = np.uint32(24)
    I16 = np.uint32(25)
    I32 = np.uint32(26)
    I64 = np.uint32(27)
    F64 = np.uint32(28)
    IQ1_M = np.uint32(29)


class GGUFMetadataValueType(Enum):
    UINT8 = np.uint32(0)
    INT8 = np.uint32(1)
    UINT16 = np.uint32(2)
    INT16 = np.uint32(3)
    UINT32 = np.uint32(4)
    INT32 = np.uint32(5)
    FLOAT32 = np.uint32(6)
    BOOL = np.uint32(7)
    STRING = np.uint32(8)
    ARRAY = np.uint32(9)
    UINT64 = np.uint32(10)
    INT64 = np.uint32(11)
    FLOAT64 = np.uint32(12)


# NOTE: we set bool in the set.
GGUF_METADATA_TYPR_NUMBER_SET = {GGUFMetadataValueType.UINT8, GGUFMetadataValueType.INT8,
                                 GGUFMetadataValueType.UINT16, GGUFMetadataValueType.INT16,
                                 GGUFMetadataValueType.UINT32, GGUFMetadataValueType.INT32,
                                 GGUFMetadataValueType.FLOAT32, GGUFMetadataValueType.UINT64,
                                 GGUFMetadataValueType.INT64, GGUFMetadataValueType.FLOAT64,
                                 GGUFMetadataValueType.BOOL}


class FormatCharacter:
    UINT8 = "B"
    INT8 = "b"
    UINT16 = "H"
    INT16 = "h"
    UINT32 = "I"
    INT32 = "i"
    UINT64 = "Q"
    INT64 = "q"
    FLOAT32 = "f"
    FLOAT64 = "d"
    FLOAT16 = "e"
    BOOL = "?"


ggml_type_np_type_dict = {
    GGMLType.F32: FormatCharacter.FLOAT32, GGMLType.F16: FormatCharacter.FLOAT16
}

FORMAT_CHARACTER_DICT = {
    "UINT8": "B",
    "INT8": "b",
    "UINT16": "H",
    "INT16": "h",
    "UINT32": "I",
    "INT32": "i",
    "UINT64": "Q",
    "INT64": "q",
    "FLOAT32": "f",
    "FLOAT64": "d",
    "BOOL": "?"
}

FORMAT_NP_TYPE_DICT = {
    "B": np.uint8,
    "b": np.int8,
    "H": np.uint16,
    "h": np.int16,
    "I": np.uint32,
    "i": np.int32,
    "Q": np.uint64,
    "q": np.int64,
    "f": np.float32,
    "d": np.float64,
    "?": np.bool_
}


class GGUFString:
    def __init__(self, length: np.uint64, string: str):
        """
        :param length: The length of the string, in bytes. -- UINT64
        :param string: The string as a UTF-8 non-null-terminated string.
        """
        self.length = length
        self.string = string

    def __str__(self):
        return self.string


T = TypeVar("T", np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.float32, np.uint64, np.int64,
            np.float64, np.bool_, GGUFString)

K = TypeVar("K", np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.float32, np.uint64, np.int64,
            np.float64)

V = TypeVar("V", np.float32, np.float16, np.uint8)


class GGUFMetadataValue(metaclass=abc.ABCMeta):
    def __init__(self, value: Any):
        self.value = value

    def __str__(self):
        return "{0}".format(self.value)


class GGUFMetadataValueT(GGUFMetadataValue):
    def __init__(self, value: T):
        self.value = value


class GGUFMetadataValueGGUFArray(GGUFMetadataValue):
    def __init__(self, type_: GGUFMetadataValueType, length: np.uint64, value: List[GGUFMetadataValue]):
        """
        :param type_: Any value type is valid, including arrays.
        :param length: UINT64.
        :param value: The array of values.
        """
        self.type = type_
        self.length = length
        self.value = value


class GGUFMetadataKV:
    def __init__(self, key: GGUFString, value_type: GGUFMetadataValueType, value: GGUFMetadataValue):
        self.key = key
        self.value_type = value_type
        self.value = value

    def __str__(self):
        return "key: {}, value_type: {}, value: {}".format(str(self.key), str(self.value_type), str(self.value))


class GGUFHeader:
    def __init__(self, magic: np.uint32, version: np.uint32, tensor_count: np.uint64, metadata_kv_count: np.uint64,
                 metadata_kv: List[GGUFMetadataKV]):
        """
        :param magic: UINT32.
        :param version: UINT32.
        :param tensor_count: UINT64.
        :param metadata_kv_count: UINT64.
        :param metadata_kv: List[GGUFMetadataKV]
        """
        self.magic = magic
        self.version = version
        self.tensor_count = tensor_count
        self.meta_kv_count = metadata_kv_count
        self.metadata_kv = metadata_kv


class GGUFTensorInfo:
    def __init__(self, name: GGUFString, n_dimensions: np.uint32, dimensions: [np.uint64], type_: GGMLType,
                 offset: np.uint64):
        """
        :param name:
        :param n_dimensions: UINT32
        :param dimensions: UINT64
        :param type_:
        :param offset: UINT64
        """
        self.name = name
        self.n_dimensions = n_dimensions
        self.dimensions = dimensions
        self.type = type_
        self.offset = offset

    def __str__(self):
        return "name: {0}, n_dimensions: {1}, dimensions: {2}, type: {3}, offset: {4}".format(str(self.name),
                                                                                              self.n_dimensions,
                                                                                              self.dimensions,
                                                                                              str(self.type),
                                                                                              self.offset)


class GGUFFile:
    def __init__(self, header: GGUFHeader, tensor_infos: List[GGUFTensorInfo], padding: List[np.uint8],
                 tensor_data: List[np.uint8]):
        """
        :param header:
        :param tensor_infos:
        :param padding: UINT8
        :param tensor_data: UINT8
        """
        self.header = header
        self.tensor_infos = tensor_infos
        self.padding = padding
        self.tensor_data = tensor_data


QK_K = 256
# refer to gguf package, value is (block size, type size)
GGML_QUANT_SIZES_DICT = {
    GGMLType.F32:  (1, 4),
    GGMLType.F16:  (1, 2),
    GGMLType.Q4_0: (32, 2 + 16),
    GGMLType.Q4_1: (32, 2 + 2 + 16),
    GGMLType.Q5_0: (32, 2 + 4 + 16),
    GGMLType.Q5_1: (32, 2 + 2 + 4 + 16),
    GGMLType.Q8_0: (32, 2 + 32),
    GGMLType.Q8_1: (32, 4 + 4 + 32),
    GGMLType.Q2_K: (256, 2 + 2 + QK_K // 16 + QK_K // 4),
    GGMLType.Q3_K: (256, 2 + QK_K // 4 + QK_K // 8 + 12),
    GGMLType.Q4_K: (256, 2 + 2 + QK_K // 2 + 12),
    GGMLType.Q5_K: (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    GGMLType.Q6_K: (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    GGMLType.Q8_K: (256, 4 + QK_K + QK_K // 8),
}
