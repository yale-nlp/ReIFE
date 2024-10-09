from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
)
from ..base_llm import BaseLLM, BaseVLLM
import torch
from .registry import register_model


@register_model("llama2")
class Llama2(BaseLLM):
    def __init__(
        self,
        model_pt="meta-llama/Llama-2-7b-chat-hf",
        device="auto",
        pad_token="<pad>",
        system_prompt: str | None = None,
    ):
        """
        Initializes the Llama2 model.

        Args:
            model_pt (str, optional): The path or name of the pre-trained model. Defaults to "meta-llama/Llama-2-7b-chat-hf".
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
        self.max_len = 4096
        self.B_SYS = "<<SYS>>\n"
        self.E_SYS = "\n<</SYS>>\n\n"
        self.B_INST = "[INST]"
        self.E_INST = "[/INST]"
        if system_prompt is not None:
            self.DEFAULT_SYSTEM_PROMPT = system_prompt
        else:
            self.DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

    def get_scoring_prompt(self, dialog):
        def _concat_messages(messages):
            if messages[0]["role"] != "system":
                messages = [
                    {
                        "role": "system",
                        "content": self.DEFAULT_SYSTEM_PROMPT,
                    }
                ] + messages
            messages = [
                {
                    "role": messages[1]["role"],
                    "content": self.B_SYS
                    + messages[0]["content"]
                    + self.E_SYS
                    + messages[1]["content"],
                }
            ] + messages[2:]
            assert all([msg["role"] == "user" for msg in messages[::2]]) and all(
                [msg["role"] == "assistant" for msg in messages[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: list[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{self.B_INST} {(prompt['content']).strip()} {self.E_INST} {(answer['content']).strip()}",
                    )
                    for prompt, answer in zip(
                        messages[::2],
                        messages[1::2],
                    )
                ],
                [],
            )
            return dialog_tokens

        assert dialog[-1]["role"] == "assistant"  # last message should be assistant
        tokenized_example = _concat_messages(dialog).flatten()
        if len(tokenized_example) > self.max_len:
            print("Warning: input length exceeds max_len")
        input_ids = tokenized_example[: self.max_len]

        prompt_example = _concat_messages(
            dialog[:-1] + {"role": "assistant", "content": ""}
        ).flatten()
        prompt_input_ids = prompt_example[: self.max_len]
        return input_ids, (len(prompt_input_ids), len(input_ids))


@register_model("llama2vllm")
class Llama2VLLM(BaseVLLM):
    """
    A class for the Llama2 VLLM model.
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
    B_SYS = "<<SYS>>\n"
    E_SYS = "\n<</SYS>>\n\n"
    B_INST = "[INST]"
    E_INST = "[/INST]"

    def get_generation_prompt(
        self, dialog: list[dict], max_input_len: int | None = None
    ) -> list[int]:
        if dialog[0]["role"] != "system":
            dialog = [
                {
                    "role": "system",
                    "content": self.DEFAULT_SYSTEM_PROMPT,
                }
            ] + dialog
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": self.B_SYS
                + dialog[0]["content"]
                + self.E_SYS
                + dialog[1]["content"],
            }
        ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        dialog_tokens: list[int] = sum(
            [
                self.tokenizer.encode(
                    f"{self.B_INST} {(prompt['content']).strip()} {self.E_INST} {(answer['content']).strip()} ",
                )
                + [self.tokenizer.eos_token_id]
                for prompt, answer in zip(
                    dialog[::2],
                    dialog[1::2],
                )
            ],
            [],
        )
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += self.tokenizer.encode(
            f"{self.B_INST} {(dialog[-1]['content']).strip()} {self.E_INST}",
        )
        if max_input_len is None:
            max_input_len = self.max_input_len
        if len(dialog_tokens) > max_input_len:
            print(
                f"Warning: input length {len(dialog_tokens)} exceeds max input length {max_input_len}"
            )
            dialog_tokens = dialog_tokens[:max_input_len]
        return dialog_tokens
