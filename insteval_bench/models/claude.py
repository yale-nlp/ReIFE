import os
from ..base_llm import BaseLLMAPI
from .registry import register_model

model2bedrock = {
    "claude-2.1": "anthropic.claude-v2:1",
    "claude-instant-1.2": "anthropic.claude-instant-v1",
    "claude-3-sonnet-20240229": "anthropic.claude-3-sonnet-20240229-v1:0",
    "claude-3-haiku-20240307": "anthropic.claude-3-haiku-20240307-v1:0",
    "claude-3-opus-20240229": "anthropic.claude-3-opus-20240229-v1:0",
}


@register_model("claude")
class CLAUDE(BaseLLMAPI):
    def __init__(
        self,
        model_pt: str,
        key_path: str,
        account_path: str,
        parallel_size: int,
        max_retries: int = 10,
        initial_wait_time: int = 2,
        end_wait_time: int = 0,
    ):
        """
        Initializes the BaseLLMAPI object for calling API services.

        Args:
            model_pt (str): Model name
            key_path (str): Path to the key file (should be kept secret)
            account_path (str): Path to the account file (should be kept secret)
            parallel_size (int): Number of parallel processes
            max_retries (int, optional): Maximum number of retries. Defaults to 10.
            initial_wait_time (int, optional): Initial wait time. Defaults to 2.
        """

        self.model_pt = model_pt
        self.parallel_size = parallel_size
        self.initial_wait_time = initial_wait_time
        # print(account_path, key_path)
        os.environ["AWS_ACCESS_KEY_ID"] = open(account_path).read().strip()
        os.environ["AWS_SECRET_ACCESS_KEY"] = open(key_path).read().strip()

        self.max_retries = max_retries
        self.end_wait_time = end_wait_time
        from anthropic import AnthropicBedrock

        self.bedrock = AnthropicBedrock(aws_region="us-west-2")

    def _get_response(
        self,
        prompt: list[dict],
        n: int = 1,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        logprobs: int | None = None,
    ) -> list[dict]:
        """
        Get the response from the API service.

        Args:
            prompt (list[dict]): The prompt to be sent to the API service.
            n (int, optional): Number of text generations per prompt. Defaults to 1.
            max_tokens (int, optional): Maximum number of tokens in the generated text. Defaults to 1024.
            temperature (float, optional): Controls the randomness of the generated text. Higher values make the text more random. Defaults to 1.0.
            top_p (float, optional): Controls the diversity of the generated text. Lower values make the text more focused. Defaults to 1.0.
            logprobs (int | None, optional): Number of log probabilities to include in the generated text. Defaults to None.

        Returns:
            list[dict]: The response from the API service. Each response is a dictionary containing the generated text ("text"), log probabilities ("logprobs", optional), and tokens ("tokens", optional).
        """
        if n > 1:
            print(
                "Warning: CLAUDE does not support multiple generations per prompt. Using n=1."
            )
        if logprobs is not None:
            print(
                "Warning: CLAUDE does not support log probabilities. Ignoring logprobs."
            )
        if prompt[-1]["role"] != "user":
            raise ValueError("Last message should be user")
        if len([x for x in prompt if x["role"] == "user"]) > 1:
            raise ValueError("Only single round of conversation is supported")
        prompt = prompt[-1]["content"]
        try:
            api_model_key = model2bedrock.get(self.model_pt, None)
            response_obj = self.bedrock.messages.create(
                messages=[{"role": "user", "content": prompt}],
                model=api_model_key,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response_text = response_obj.content[0].text
        except Exception as e:
            print(e)
            raise
        return [{"text": response_text}]
