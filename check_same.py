from read_gguf import GGUFLoader

gguf_loader = GGUFLoader("llama-2-7b.Q2_K.gguf")
gguf_loader2 = GGUFLoader("example.gguf")

gguf_loader.load_and_print()
gguf_loader2.load_and_print()

gguf_tensor = gguf_loader.tensor_infos
gguf_tensor2 = gguf_loader2.tensor_infos

metadata = gguf_loader.metadata
metadata2 = gguf_loader2.metadata

meta_data_dict = {metadata[i].key.string: (metadata[i].value, metadata[i].value_type.name, metadata[i].value_type.value) for i in range(len(metadata))}
meta_data_dict2 = {metadata2[i].key.string: (metadata2[i].value, metadata2[i].value_type.name, metadata2[i].value_type.value) for i in range(len(metadata2))}

gguf_tensor_set = list(set(gguf_tensor))
gguf_tensor_set2 = list(set(gguf_tensor2))

gguf_tensors_name_dimension_dict = {gguf_tensor_set[i].name.string: gguf_tensor_set[i].dimensions for i in range(len(gguf_tensor_set))}
gguf_tensors_name_dimension_dict2 = {gguf_tensor_set2[i].name.string: gguf_tensor_set2[i].dimensions for i in range(len(gguf_tensor_set2))}

# 比较tensor信息是否一致
result1 = []
for key in gguf_tensors_name_dimension_dict:
    result1.append(gguf_tensors_name_dimension_dict[key] == gguf_tensors_name_dimension_dict2[key])

# 比较 metadta 是否一致
result2 = []
for key in meta_data_dict:
    result2.append(meta_data_dict[key] == meta_data_dict2[key])

print(any(result1))
print(any(result2))

print(gguf_loader.metadata_kv_count)
print(gguf_loader2.metadata_kv_count)

# 比较 tensor 的 offset 是否一致
result3 = []
for i, elem in enumerate(gguf_loader2.tensor_infos):
    print(elem.name, elem.type, elem.offset)

