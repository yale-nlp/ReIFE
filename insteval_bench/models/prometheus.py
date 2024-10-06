from ..base_llm import BaseVLLM
from .registry import register_model
from .conversation import get_conv_template


@register_model("prometheusvllm")
class PrometheusVLLM(BaseVLLM):
    """
    A class for the Prometheus VLLM model.
    """
    def get_generation_prompt(
        self, dialog: list[dict], max_input_len: int | None = None
    ) -> list[int]:
        conv = get_conv_template("mistral")

        for message in dialog:
            if message["role"] == "system":
                conv.set_system_message(message["content"])
            elif message["role"] == "user":
                conv.append_message(conv.roles[0], message["content"])

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        dialog_tokens = self.tokenizer.encode(prompt)
        if max_input_len is None:
            max_input_len = self.max_input_len
        if len(dialog_tokens) > max_input_len:
            print(
                f"Warning: input length {len(dialog_tokens)} exceeds max input length {max_input_len}"
            )
            dialog_tokens = dialog_tokens[:max_input_len]
        return dialog_tokens
