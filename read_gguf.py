import struct
from typing import List, BinaryIO

import numpy as np

from constant import FormatCharacter, GGUFException, GGUFTensorInfo, GGUFMetadataKV, GGUFString, \
    GGUFMetadataValueType, GGUF_METADATA_TYPR_NUMBER_SET, FORMAT_CHARACTER_DICT, FORMAT_NP_TYPE_DICT, K, GGMLType, V, \
    GGML_QUANT_SIZES_DICT

VALID_MAGIC_NUMBER = b"GGUF"
VALID_GGUF_VERSION = 3
ENCODING = "utf-8"


class GGUFLoader:
    def __init__(self, gguf_file_path: str):
        """
        :param gguf_file_path: gguf_file_path
        """
        self.gguf_file_path: str = gguf_file_path
        self.tensor_count: np.uint64 = np.uint64(0)
        self.metadata_kv_count: np.uint64 = np.uint(64)
        self.metadata: List[GGUFMetadataKV] = []
        self.tensor_infos: List[GGUFTensorInfo] = []
        self.f: BinaryIO = None
        self.alignment: int = 32
        self.tensors: List[List[V]] = []

    @staticmethod
    def auto_struct_unpack(f: BinaryIO, format_: str) -> K:
        standard_size = struct.calcsize(format_)
        value = FORMAT_NP_TYPE_DICT[format_](struct.unpack(format_, f.read(standard_size))[0])
        return value

    @staticmethod
    def get_gguf_string(f: BinaryIO) -> GGUFString:
        length = GGUFLoader.auto_struct_unpack(f, FormatCharacter.UINT64)
        string = f.read(length).decode(ENCODING)
        return GGUFString(length=length, string=string)

    @staticmethod
    def get_gguf_metadata_value_type(f: BinaryIO) -> GGUFMetadataValueType:
        temp_value_type = GGUFLoader.auto_struct_unpack(f, FormatCharacter.UINT32)
        return GGUFMetadataValueType(temp_value_type)

    @staticmethod
    def get_gguf_metadata_value(f: BinaryIO, value_type: GGUFMetadataValueType):
        if value_type in GGUF_METADATA_TYPR_NUMBER_SET:
            return GGUFLoader.auto_struct_unpack(f, FORMAT_CHARACTER_DICT[value_type.name])
        if value_type == GGUFMetadataValueType.STRING:
            return f.read(struct.unpack(FormatCharacter.UINT64, f.read(8))[0]).decode(ENCODING)
        if value_type == GGUFMetadataValueType.ARRAY:
            array_value_type = GGUFMetadataValueType(GGUFLoader.auto_struct_unpack(f, FormatCharacter.UINT32))
            array_length = GGUFLoader.auto_struct_unpack(f, FormatCharacter.UINT64)
            return [GGUFLoader.get_gguf_metadata_value(f, array_value_type) for _ in range(array_length)]
        raise GGUFException("unexpected metadata value type.")

    @staticmethod
    def padding(cur_offset: int, alignment: int) -> int:
        """adjust offset"""
        return np.uint64(cur_offset + (alignment - (cur_offset % alignment)) % alignment)

    def _set_up(self):
        """
        setUp: open file
        :return:
        """
        self.f = open(self.gguf_file_path, "rb")

    def _tear_down(self):
        """
        tearDown: close file
        :return:
        """
        self.f.close()

    def _check_magic_number(self):
        """
        check whether the magic number is right.
        :return: void
        """
        if self.f.read(4) != VALID_MAGIC_NUMBER:
            raise GGUFException("Invalid magic number")

    def _check_version(self):
        """
        check whether the version is 3.
        :return: void
        """
        if GGUFLoader.auto_struct_unpack(self.f, FormatCharacter.UINT32) != VALID_GGUF_VERSION:
            raise GGUFException("Invalid version")

    def _read_tensor_count(self):
        """
        read tensor numbers
        :return: void
        """
        self.tensor_count = GGUFLoader.auto_struct_unpack(self.f, FormatCharacter.UINT64)

    def _read_metadata_kv_count(self):
        """
        read meta data key-value numbers
        :return: void
        """
        self.metadata_kv_count = GGUFLoader.auto_struct_unpack(self.f, FormatCharacter.UINT64)

    def _read_metadata_key_value_pairs(self):
        """
        read metadata value.
        :return:
        """
        for i in range(self.metadata_kv_count):
            key = GGUFLoader.get_gguf_string(self.f)
            value_type = GGUFLoader.get_gguf_metadata_value_type(self.f)
            value = GGUFLoader.get_gguf_metadata_value(self.f, value_type)
            metadata_kv = GGUFMetadataKV(key, value_type, value)
            self.metadata.append(metadata_kv)

    def _read_tensors_info(self):
        """
        get tensors info
        :return:
        """
        for i in range(self.tensor_count):
            name = GGUFLoader.get_gguf_string(self.f)
            n_dimensions = GGUFLoader.auto_struct_unpack(self.f, FormatCharacter.UINT32)
            dimensions = [GGUFLoader.auto_struct_unpack(self.f, FormatCharacter.UINT64) for _ in range(n_dimensions)]
            type_ = GGMLType(GGUFLoader.auto_struct_unpack(self.f, FormatCharacter.UINT32))
            offset = GGUFLoader.auto_struct_unpack(self.f, FormatCharacter.UINT64)
            self.tensor_infos.append(GGUFTensorInfo(name, n_dimensions, dimensions, type_, offset))

    def _adjust_tensors_info(self, adjust_offset: int):
        """adjust tensors info offset"""
        for i in range(len(self.tensor_infos)):
            self.tensor_infos[i].offset += np.uint64(adjust_offset)

    def _read_tensors(self):
        """using tensors info to read tensors"""
        for tensor_info in self.tensor_infos:
            start_offset = tensor_info.offset
            # first we should move to the start_offset
            if tensor_info.offset >= self.f.tell():
                self.f.read(np.uint64(start_offset - self.f.tell()))
                # read tensors
                tensor_length = np.uint64(np.prod(tensor_info.dimensions))
                block_size, type_size = GGML_QUANT_SIZES_DICT[tensor_info.type]
                n_bytes = tensor_length * type_size // block_size
                temp_tensor = []
                if tensor_info.type == GGMLType.F32:
                    for _ in range(tensor_length):
                        temp_tensor.append(GGUFLoader.auto_struct_unpack(self.f, FormatCharacter.FLOAT32))
                elif tensor_info.type == GGMLType.F16:
                    for _ in range(tensor_length):
                        temp_tensor.append(np.frombuffer(self.f.read(2), dtype=np.float16)[0])
                else:
                    for _ in range(int(n_bytes)):
                        temp_tensor.append(GGUFLoader.auto_struct_unpack(self.f, "B"))
                self.tensors.append(temp_tensor)

    def load_and_print(self):
        """
        main function to load GGUF info and print summary
        :return: void
        """
        self._set_up()
        try:
            # CAUTION: these methods should execute by ORDER!
            self._check_magic_number()
            self._check_version()
            self._read_tensor_count()
            self._read_metadata_kv_count()
            self._read_metadata_key_value_pairs()
            self._read_tensors_info()
            self._adjust_tensors_info(GGUFLoader.padding(self.f.tell(), self.alignment))
            self._read_tensors()
        except GGUFException as e:
            print(e)
            self._tear_down()
        finally:
            self._tear_down()


if __name__ == "__main__":
    path = "qwen1_5-0_5b-chat-q2_k.gguf"
    gguf_loader = GGUFLoader(path)
    gguf_loader.load_and_print()
