from ..base_llm import BaseVLLM
from .registry import register_model
from vllm import SamplingParams
from tqdm import tqdm
import vllm

vllm_version = vllm.__version__

@register_model("glmvllm")
class GLMVLLM(BaseVLLM):
    """
    A class for the GLM VLLM model without system prompt.
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
    
    def generate(
        self,
        prompts: list[list[dict]],
        n: int = 1,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        logprobs: int | None = None,
        use_tqdm: bool = True,
        input_length: int | None = None,
    ) -> list[list[dict]]:
        """
        Generates text based on the given prompts using the language model.

        Args:
            prompts (list[list[dict]]): List of prompts, where each prompt is a list of dictionaries.
            n (int, optional): Number of text generations per prompt. Defaults to 1.
            max_tokens (int, optional): Maximum number of tokens in the generated text. Defaults to 1024.
            temperature (float, optional): Controls the randomness of the generated text. Higher values make the text more random. Defaults to 1.0.
            top_p (float, optional): Controls the diversity of the generated text. Lower values make the text more focused. Defaults to 1.0.
            logprobs (int | None, optional): Number of log probabilities to include in the generated text. Defaults to None.
            use_tqdm (bool, optional): Whether to display a progress bar. Defaults to True.
            input_length (int | None, optional): The length of the input. If not provided, the default model input length will be used. Defaults to None.

        Returns:
            list[list[dict]]: List of generated text, where each generated text is a list of dictionaries containing the generated text, log probabilities, and tokens.
        """

        max_input_len = self.max_input_len if input_length is None else input_length

        if max_tokens + max_input_len > self.max_model_len:
            raise ValueError(
                f"max_tokens ({max_tokens}) + max_input_len ({max_input_len}) > max_model_len ({self.max_model_len})"
            )

        prompts = [
            self.get_generation_prompt(prompt, input_length)
            for prompt in tqdm(prompts, desc="preparing prompts", disable=not use_tqdm)
        ]

        outputs = self.model.generate(
            prompt_token_ids=prompts,
            sampling_params=SamplingParams(
                n=n,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                logprobs=logprobs,
                stop_token_ids=self.STOP_TOKEN_IDS,
            ),
            use_tqdm=use_tqdm,
        )
        _outputs = []
        
        def decode(x):
            x = self.tokenizer.convert_ids_to_tokens(x)
            x = self.tokenizer.convert_tokens_to_string([x])
            return x

        for output in outputs:
            output = output.outputs
            _output = []
            for x in output:
                if logprobs is not None:
                    _logprobs = []
                    for y in x.logprobs:
                        if vllm_version >= "0.4.0":
                            _logprobs.append(
                                {decode(k): v.logprob for k, v in y.items()}
                            )
                        else:
                            _logprobs.append(
                                {decode(k): v for k, v in y.items()}
                            )
                else:
                    _logprobs = None
                tokens = [decode(y) for y in x.token_ids]
                text = x.text
                if self.STOP_TOKEN_IDS is not None:
                    for stop_token_id in self.STOP_TOKEN_IDS:
                        stop_token = decode(stop_token_id)
                        if text.endswith(stop_token):
                            text = text[: -len(stop_token)]
                            break
                _output.append(
                    {
                        "text": text,
                        "logprobs": _logprobs,
                        "tokens": tokens,
                    }
                )
            _outputs.append(_output)
        return _outputs