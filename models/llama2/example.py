#!/usr/bin/env python3
import json
import logging
import sys
from pathlib import Path

from constant import GGML_TENSOR_QUANTIZE_DICT
from models.llama2.utils import MsCkptRefactorHelper

# Necessary to load the local gguf package
sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf import GGUFWriter  # noqa: E402


class Llama2Writer:
    def __init__(self, metadata_json_path: str, tensor_ggml_types_json_path: str,
                 layer_name_map_json_path: str, ckpt_file_path: str):
        self.metadata_json_path = metadata_json_path
        self.tensor_ggml_types_json_path = tensor_ggml_types_json_path
        self.layer_name_map_json_path = layer_name_map_json_path
        self.ckpt_file_path = ckpt_file_path
        # ms ckpt helper
        self.ms_helper: MsCkptRefactorHelper
        # gguf_metadata kv pairs
        self.metadata_kv_pairs: dict = {}
        # gguf writer
        self.gguf_writer: GGUFWriter
        # tensor_ggml_types_map
        self.tensor_ggml_type_dict: dict = {}

    def __set_up(self):
        # init ms helper
        self.ms_helper = MsCkptRefactorHelper(self.ckpt_file_path, self.layer_name_map_json_path)
        self.ms_helper.do_refactor()
        # init metadata kv pairs
        with open(self.metadata_json_path, "r", encoding="utf-8") as f:
            self.metadata_kv_pairs = json.load(f)
        # init tensor_ggml_types_dict
        with open(self.tensor_ggml_types_json_path, "r", encoding="utf-8") as g:
            self.tensor_ggml_type_dict = json.load(g)
        # init gguf writer
        self.gguf_writer = GGUFWriter("example.gguf", "llama")

    def __write_metadata(self):
        for metadata_key in self.metadata_kv_pairs:
            if metadata_key == "general.architecture":
                # we have set the arch in __set_up method
                continue
            if isinstance(self.metadata_kv_pairs[metadata_key], int):
                self.gguf_writer.add_uint32(metadata_key, self.metadata_kv_pairs[metadata_key])
            elif isinstance(self.metadata_kv_pairs[metadata_key], float):
                self.gguf_writer.add_float32(metadata_key, self.metadata_kv_pairs[metadata_key])
            elif isinstance(self.metadata_kv_pairs[metadata_key], str):
                self.gguf_writer.add_string(metadata_key, self.metadata_kv_pairs[metadata_key])
            elif isinstance(self.metadata_kv_pairs[metadata_key], list):
                self.gguf_writer.add_array(metadata_key, self.metadata_kv_pairs[metadata_key])
            else:
                logging.error("Unexpected metadata key type: {0} of key :{1}".format(type(
                    self.metadata_kv_pairs[metadata_key]), metadata_key))

    def __write_tensors(self):
        for tensor_name in self.ms_helper.ckpt_dict:
            # convert ms tensor to ndarray
            ndarray_tensor = MsCkptRefactorHelper.convert_ms_tensor_to_ndarray(
                self.ms_helper.ckpt_dict[tensor_name], tensor_name)
            # write
            logging.info("ndarray tensor type: {0}".format(ndarray_tensor.dtype))
            raw_type = GGML_TENSOR_QUANTIZE_DICT[self.tensor_ggml_type_dict[tensor_name]]
            self.gguf_writer.add_tensor(tensor_name, ndarray_tensor, raw_dtype=raw_type)

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
    writer = Llama2Writer(metadata_json_path="llama2-7b-gguf-metadata.json",
                          tensor_ggml_types_json_path="llama2_7b_Q2_K_tensor_ggml_typs.json",
                          layer_name_map_json_path="llama2_layer_name_map.json",
                          ckpt_file_path="llama2_7b.ckpt")
    writer.write()

