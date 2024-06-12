import json
import logging

from constant import GGUFMetadataValueType
from read_gguf import GGUFLoader


class MetadataDumpHelper:
    def __init__(self, origin_gguf_file_path: str, metadata_json_output_path: str):
        self.origin_gguf_file_path = origin_gguf_file_path
        self.metadata_json_output_path = metadata_json_output_path
        self.gguf_loader: GGUFLoader
        self.meta_data_dict: dict = {}

    def __set_up(self):
        self.gguf_loader = GGUFLoader(self.origin_gguf_file_path)
        self.gguf_loader.load_and_print()

    def __get_metadata_dict(self):
        for metadata in self.gguf_loader.metadata:
            if metadata.value_type == GGUFMetadataValueType.UINT32:
                self.meta_data_dict[metadata.key.string] = int(metadata.value)
            elif metadata.value_type == GGUFMetadataValueType.FLOAT32:
                self.meta_data_dict[metadata.key.string] = float(metadata.value)
            elif metadata.value_type == GGUFMetadataValueType.STRING:
                self.meta_data_dict[metadata.key.string] = str(metadata.value)
            elif metadata.value_type == GGUFMetadataValueType.ARRAY:
                self.meta_data_dict[metadata.key.string] = GGUFLoader.convert_gguf_metadata_array_to_list(metadata,
                                                                                                          metadata.key.string)
            else:
                logging.error("unsupport data type {0}", metadata.value_type)
        with open(self.metadata_json_output_path, "w+", encoding="utf-8") as f:
            json.dump(self.meta_data_dict, f, ensure_ascii=False)

    def dump_json_file(self):
        self.__set_up()
        self.__get_metadata_dict()


if __name__ == "__main__":
    # example
    metadata_json_dump_helper = MetadataDumpHelper("llama2/llama-2-7b.Q2_K.gguf",
                                                   "llama2/configs/llama2-7b-gguf-metadata.json")
    metadata_json_dump_helper.dump_json_file()
