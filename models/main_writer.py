#!/usr/bin/env python3
import json
import logging
import sys
from pathlib import Path

from models.ckpt_convert_util import MsCkptRefactorHelper

# Necessary to load the local gguf package
sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf import GGUFWriter, GGMLQuantizationType  # noqa: E402


class Writer:
    def __init__(self, metadata_json_path: str, layer_name_map_json_path: str,
                 ckpt_file_path: str, arch: str, need_transpose: bool = False):
        """
        :param metadata_json_path: metadata_kv_pairs json file path
        :param layer_name_map_json_path: layer name map json file path
        :param ckpt_file_path: MindSpore json file path
        :param arch: model arch, such as "baichuan", "llama"
        :param need_transpose: whether you need transpose, default False
        """
        self.metadata_json_path = metadata_json_path
        self.layer_name_map_json_path = layer_name_map_json_path
        self.ckpt_file_path = ckpt_file_path
        # ms ckpt helper
        self.ms_helper: MsCkptRefactorHelper
        # gguf_metadata kv pairs
        self.metadata_kv_pairs: dict = {}
        # gguf writer
        self.gguf_writer: GGUFWriter
        # arch
        self.arch = arch
        # whether you need transpose
        self.transpose = need_transpose

    def __set_up(self):
        # init ms helper
        self.ms_helper = MsCkptRefactorHelper(self.ckpt_file_path, self.layer_name_map_json_path, self.transpose)
        self.ms_helper.do_refactor()
        # init metadata kv pairs
        with open(self.metadata_json_path, "r", encoding="utf-8") as f:
            self.metadata_kv_pairs = json.load(f)
        # init gguf writer
        self.gguf_writer = GGUFWriter("example.gguf", self.arch)

    def __write_metadata(self):
        for metadata_key in self.metadata_kv_pairs:
            if metadata_key == "general.architecture":
                # we have set the arch in __set_up method
                continue
            # we should check if it is bool first, because isinstance(True, int) is also True
            if isinstance(self.metadata_kv_pairs[metadata_key], bool):
                self.gguf_writer.add_bool(metadata_key, self.metadata_kv_pairs[metadata_key])
            elif isinstance(self.metadata_kv_pairs[metadata_key], int):
                self.gguf_writer.add_uint32(metadata_key, self.metadata_kv_pairs[metadata_key])
            elif isinstance(self.metadata_kv_pairs[metadata_key], float):
                self.gguf_writer.add_float32(metadata_key, self.metadata_kv_pairs[metadata_key])
            elif isinstance(self.metadata_kv_pairs[metadata_key], str):
                self.gguf_writer.add_string(metadata_key, self.metadata_kv_pairs[metadata_key])
            elif isinstance(self.metadata_kv_pairs[metadata_key], list):
                self.gguf_writer.add_array(metadata_key, self.metadata_kv_pairs[metadata_key])
            else:
                logging.error("Unexpected metadata key type: {0} of key :{1}", type(
                    self.metadata_kv_pairs[metadata_key]), metadata_key)

    def __write_tensors(self):
        for tensor_name in self.ms_helper.ckpt_dict:
            # convert ms tensor to ndarray
            ndarray_tensor = MsCkptRefactorHelper.convert_ms_tensor_to_ndarray(
                self.ms_helper.ckpt_dict[tensor_name], tensor_name)
            # write
            logging.info("ndarray tensor type: {0}", ndarray_tensor.dtype)
            self.gguf_writer.add_tensor(tensor_name, ndarray_tensor, raw_dtype=GGMLQuantizationType.F32)

    def __tear_down(self):
        self.gguf_writer.write_header_to_file()
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file()
        self.gguf_writer.close()

    def write(self):
        self.__set_up()
        self.__write_metadata()
        self.__write_tensors()
        self.__tear_down()


if __name__ == '__main__':
    writer = Writer(metadata_json_path="llama2/configs/llama2-7b-gguf-metadata.json",
                    layer_name_map_json_path="llama2/configs/llama2_layer_name_map.json",
                    ckpt_file_path="llama2/llama2_7b.ckpt",
                    arch="llama",
                    need_transpose=False)
    writer.write()
