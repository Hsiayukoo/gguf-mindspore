from read_gguf import GGUFLoader

loader = GGUFLoader("example.gguf")
loader.load_and_print()
tensor_name_type_dict = {elem.name.string: (elem.type, elem.offset) for elem in loader.tensor_infos}

for tensor in loader.tensor_infos:
    print(tensor.name.string, tensor_name_type_dict[tensor.name.string])

loader = GGUFLoader("example.gguf", True)
loader.load_and_print()

