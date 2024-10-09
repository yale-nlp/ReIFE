from ..base_llm import BaseVLLM
from .registry import register_model


@register_model("hfvllm")
class HFVLLM(BaseVLLM):
    """
    A class for the base HF VLLM model.
    """
    def get_generation_prompt(
        self, dialog: list[dict], max_input_len: int | None = None
    ) -> list[int]:
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens = self.tokenizer.apply_chat_template(
            dialog,
            add_generation_prompt=True,
        )
        if max_input_len is None:
            max_input_len = self.max_input_len
        if len(dialog_tokens) > max_input_len:
            print(
                f"Warning: input length {len(dialog_tokens)} exceeds max input length {max_input_len}"
            )
            dialog_tokens = dialog_tokens[:max_input_len]
        return dialog_tokens


@register_model("hfnosysvllm")
class HFNoSysVLLM(BaseVLLM):
    """
    A class for the HF VLLM model without system prompt.
    """
    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

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
        dialog[1]["content"] = dialog[0]["content"].strip() + "\n" + dialog[1]["content"]
        dialog = dialog[1:]
        
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens = self.tokenizer.apply_chat_template(
            dialog,
            add_generation_prompt=True,
        )
        if max_input_len is None:
            max_input_len = self.max_input_len
        if len(dialog_tokens) > max_input_len:
            print(
                f"Warning: input length {len(dialog_tokens)} exceeds max input length {max_input_len}"
            )
            dialog_tokens = dialog_tokens[:max_input_len]
        return dialog_tokens