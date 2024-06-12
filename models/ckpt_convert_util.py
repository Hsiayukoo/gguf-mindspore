"""
Utils to load MindSpore ckpt file and convert to numpy ndArray
"""
import copy
import json
import logging

import mindspore as ms
import numpy as np
from mindspore import ops


class MsCkptRefactorHelper:

    @staticmethod
    def convert_ms_tensor_to_ndarray(ms_tensor: ms.Tensor, tensor_name: str) -> np.ndarray:
        """
        converts ms tensor to ndarray.
        :param ms_tensor:
        :param tensor_name:
        :return:
        """
        logging.info("now convert ms tensor name is: {0}".format(tensor_name))
        return ms_tensor.asnumpy().astype(np.float32)

    def __init__(self, ms_ckpt_path: str, name_map_path: str, transpose: bool = False):
        """
        :param ms_ckpt_path: ms ckpt file path
        :param name_map_path: ms to gguf layer name map path
        :param transpose: default False, means transpose the tensor
        """
        self.ckpt_dict = ms.load_checkpoint(ms_ckpt_path)
        self._name_map_path = name_map_path
        self._ms_to_gguf_map: dict = {}
        self.full_name_ms_to_gguf_map: dict = {}
        self.need_transpose = transpose

    def _read_name_map_json(self):
        with open(self._name_map_path, encoding="utf-8", mode="r") as f:
            self._ms_to_gguf_map = json.load(f)

    def _layer_rename(self):
        ckpt_layer_names = list(self.ckpt_dict.keys())
        ckpt_layer_names_copy = copy.deepcopy(ckpt_layer_names)
        for i in range(len(ckpt_layer_names)):
            for map_key in self._ms_to_gguf_map:
                if map_key in self._ms_to_gguf_map:
                    ckpt_layer_names[i] = ckpt_layer_names[i].replace(map_key, self._ms_to_gguf_map[map_key])
            self.full_name_ms_to_gguf_map[ckpt_layer_names_copy[i]] = ckpt_layer_names[i]
        for rename_layer in self.full_name_ms_to_gguf_map:
            self.ckpt_dict[self.full_name_ms_to_gguf_map[rename_layer]] = self.ckpt_dict.pop(rename_layer)

    def _layer_tensor_transpose(self):
        if self.need_transpose:
            for layer in self.ckpt_dict:
                # todo: now only support 2d
                if self.ckpt_dict[layer].dim() == 2:
                    self.ckpt_dict[layer] = ops.transpose(self.ckpt_dict[layer], (1, 0))

    def do_refactor(self):
        self._read_name_map_json()
        self._layer_rename()
        self._layer_tensor_transpose()
