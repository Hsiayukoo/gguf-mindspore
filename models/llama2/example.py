#!/usr/bin/env python3
import json
import logging
import sys
from pathlib import Path

from models.llama2.utils import MsCkptRefactorHelper
from read_gguf import GGUFLoader

# Necessary to load the local gguf package
sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf import GGUFWriter  # noqa: E402


# Example usage:
def write_base_informations() -> None:
    ms_helper = MsCkptRefactorHelper("../../llama2_7b.ckpt", "./llama2_layer_name_map.json")
    ms_helper.do_refactor()
    with open("llama2-7b-gguf-metadata.json", "r", encoding="utf-8") as f:
        metadata_dict = json.load(f)
    # Example usage with a file
    gguf_writer = GGUFWriter("../../example.gguf", "llama")
    for metadata_key in metadata_dict:
        print(metadata_key)
        if metadata_key == "general.architecture":
            print("pass 了")
            continue
        if isinstance(metadata_dict[metadata_key], int):
            gguf_writer.add_uint32(metadata_key, metadata_dict[metadata_key])
        elif isinstance(metadata_dict[metadata_key], float):
            gguf_writer.add_float32(metadata_key, metadata_dict[metadata_key])
        elif isinstance(metadata_dict[metadata_key], str):
            gguf_writer.add_string(metadata_key, metadata_dict[metadata_key])
        elif isinstance(metadata_dict[metadata_key], list):
            gguf_writer.add_array(metadata_key, metadata_dict[metadata_key])
        else:
            logging.error("Unexpected metadata key type: {0} of key :{1}".format(type(metadata_dict[metadata_key]), metadata_key))
    for tensor_name in ms_helper.ckpt_dict:
        gguf_writer.add_tensor(tensor_name, MsCkptRefactorHelper.convert_ms_tensor_to_ndarray(ms_helper.ckpt_dict[tensor_name], tensor_name))
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()


if __name__ == '__main__':
    write_base_informations()
    gguf_loader = GGUFLoader("../../example.gguf")
    gguf_loader.load_and_print()
    for tensor_info in gguf_loader.tensor_infos:
        print(tensor_info)

