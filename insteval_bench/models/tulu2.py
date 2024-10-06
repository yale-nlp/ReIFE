from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
)
from ..base_llm import BaseLLM, BaseVLLM
import torch
from .registry import register_model


@register_model("tulu2")
class Tulu2(BaseLLM):
    def __init__(
        self,
        model_pt="allenai/tulu-2-7b",
        device="auto",
        pad_token="<pad>",
    ):
        """
        Initializes the Tulu2 model.

        Args:
            model_pt (str, optional): The path or name of the pre-trained model. Defaults to "allenai/tulu-2-7b".
            device (str, optional): The device to run the model on. Defaults to "auto".
            pad_token (str, optional): The padding token. Defaults to "<pad>".
        """

        self.pad_token = pad_token
        self.device = device
        tokenizer = LlamaTokenizer.from_pretrained(model_pt, use_fast=False)
        if self.device == "auto":
            model = LlamaForCausalLM.from_pretrained(
                model_pt, torch_dtype=torch.bfloat16, device_map="auto"
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                model_pt, torch_dtype=torch.bfloat16
            ).to(device)
        # add pad token
        tokenizer.add_special_tokens({"pad_token": pad_token})
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.model.embed_tokens.padding_idx = tokenizer.pad_token_id
        model.config.vocab_size = 32008
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self.max_len = 8192

    def get_scoring_prompt(self, dialog):
        def _concat_messages(messages):
            message_text = ""
            for message in messages:
                if message["role"] == "system":
                    message_text += "<|system|>\n" + message["content"].strip() + "\n"
                elif message["role"] == "user":
                    message_text += "<|user|>\n" + message["content"].strip() + "\n"
                elif message["role"] == "assistant":
                    message_text += (
                        "<|assistant|>\n"
                        + message["content"].strip()
                        + self.tokenizer.eos_token
                        + "\n"
                    )
                else:
                    raise ValueError("Invalid role: {}".format(message["role"]))
            return message_text

        assert dialog[-1]["role"] == "assistant"  # last message should be assistant
        message_text = _concat_messages(dialog).strip()
        _tokenized_example = self.tokenizer(
            message_text,
            return_tensors="pt",
        )
        if len(_tokenized_example.input_ids[0]) > self.max_len:
            print("Warning: input length exceeds max_len")
        tokenized_example = self.tokenizer(
            message_text,
            return_tensors="pt",
            max_length=self.max_len,
            truncation=True,
        )
        input_ids = tokenized_example.input_ids.flatten()

        prompt_text = _concat_messages(dialog[:-1]) + "<|assistant|>\n"
        propmt_example = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            max_length=self.max_len,
            truncation=True,
        )
        prompt_input_ids = propmt_example.input_ids.flatten()
        return input_ids, (len(prompt_input_ids), len(input_ids))


@register_model("tulu2vllm")
class Tulu2VLLM(BaseVLLM):
    def get_generation_prompt(
        self, dialog: list[dict], max_input_len: int | None = None
    ) -> list[int]:
        message_text = ""
        for message in dialog:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += (
                    "<|assistant|>\n"
                    + message["content"].strip()
                    + self.tokenizer.eos_token
                    + "\n"
                )
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        message_text += "<|assistant|>\n"
        if max_input_len is None:
            max_input_len = self.max_input_len
        _tokenized_example = self.tokenizer(
            message_text,
        )
        if len(_tokenized_example.input_ids) > max_input_len:
            print("Warning: input length exceeds max_input_len")
        tokenized_example = self.tokenizer(
            message_text,
            max_length=max_input_len,
            truncation=True,
        )
        input_ids = tokenized_example.input_ids
        return input_ids
