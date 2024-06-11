import json

from constant import GGUFMetadataValueType
from read_gguf import GGUFLoader


a = GGUFLoader("models/llama2/llama-2-7b.Q2_K.gguf")
a.load_and_print()


meta_data_dict = {}
for meta_data in a.metadata:
    if meta_data.value_type == GGUFMetadataValueType.UINT32:
        meta_data_dict[meta_data.key.string] = int(meta_data.value)
    elif meta_data.value_type == GGUFMetadataValueType.FLOAT32:
        meta_data_dict[meta_data.key.string] = float(meta_data.value)
    elif meta_data.value_type == GGUFMetadataValueType.STRING:
        meta_data_dict[meta_data.key.string] = str(meta_data.value)
    elif meta_data.value_type == GGUFMetadataValueType.ARRAY:
        meta_data_dict[meta_data.key.string] = GGUFLoader.convert_gguf_metadata_array_to_list(meta_data, meta_data.key.string)
    else:
        print("unsupport data type", meta_data.value_type)

with open("models/llama2/llama2-7b-gguf-metadata.json", "w+", encoding="utf-8") as f:
    json.dump(meta_data_dict, f, ensure_ascii=False)
