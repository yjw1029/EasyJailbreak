import fastchat.model
from vllm import LLM, SamplingParams
import torch
import warnings
from typing import Any, Callable, Tuple, List, Optional, Union
from transformers import AutoTokenizer

from .model_base import BlackBoxModelBase


def get_compute_capability():
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this device!")

    capability_str = torch.cuda.get_device_capability()
    capability = float(f"{capability_str[0]}.{capability_str[1]}")
    return capability


def check_bf16_support():
    capability = get_compute_capability()
    if capability >= 8.0:
        return True
    return False


class vLLMModel(BlackBoxModelBase):
    def __init__(
        self,
        model_name,
        trust_remote_code=True,
        generation_config={},
        **kwargs,
    ):
        self.model_name = model_name
        self.tensor_parallel_size = torch.cuda.device_count()
        self.trust_remote_code = trust_remote_code

        self.model = self.load_model()
        self.conversation = self.get_conv_template()
        self.generation_config = generation_config

    def get_conv_template(self):
        conv_template = fastchat.model.get_conversation_template(self.model_name)
        return conv_template

    def load_model(self):
        if check_bf16_support():
            dtype = "bfloat16"
        else:
            dtype = "float16"

        self.model = LLM(
            model=self.model_name,
            trust_remote_code=self.trust_remote_code,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=dtype,
        )
        return self.model

    def load_generation_config(self):
        self.generation_config = SamplingParams(
            temperature=self.generation_config.get("temperature", 1),
            max_tokens=self.generation_config.get("max_new_tokens", 2048),
            top_p=self.generation_config.get("top_p", 1),
            stop=self.conversation.stop_str,
            stop_token_ids=self.conversation.stop_token_ids,
        )
        return self.generation_config

    def get_prompt(self, messages, clear_old_history=True, **kwargs):
        if clear_old_history:
            self.conversation.messages = []

        if isinstance(messages, str):
            messages = [messages]

        assert len(messages) % 2 == 1, "The number of messages should be even."

        for index, message in enumerate(messages):
            self.conversation.append_message(
                self.conversation.roles[index % 2], message
            )
        self.conversation.append_message(self.conversation.roles[1], None)

        return self.conversation.get_prompt()

    def generate(self, messages, clear_old_history=True, **kwargs):
        prompt = self.get_prompt(messages, clear_old_history)

        sampling_params = self.load_generation_config()
        responses = self.model.generate([prompt], sampling_params)
        responses = [output.outputs[0].text for output in responses]
        return responses[0]

    def batch_generate(self, conversations, **kwargs):
        """
        Generates responses for multiple conversations in a batch.
        :param list[list[str]]|list[str] conversations: A list of conversations, each as a list of messages.
        :return list[str]: A list of responses for each conversation.
        """
        prompts = []
        for conversation in conversations:
            if isinstance(conversation, str):
                warnings.warn(
                    "For batch generation based on several conversations, provide a list[str] for each conversation. "
                    "Using list[list[str]] will avoid this warning."
                )
            prompts.append(self.get_prompt(conversation, **kwargs))

        sampling_params = self.load_generation_config()
        responses = self.model.generate(prompts, sampling_params)
        responses = [output.outputs[0].text for output in responses]
        return responses

    def set_system_message(self, system_message: str):
        self.conversation.system_message = system_message
