from ..base_llm import BaseLLMAPI
from .registry import register_model


@register_model("gpt")
class GPT(BaseLLMAPI):
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
            end_wait_time (int, optional): End wait time. Defaults to 0.
        """
        from openai import OpenAI

        self.model_name = model_pt
        with open(key_path, "r") as f:
            api_key = f.read().strip()
        with open(account_path, "r") as f:
            organization = f.read().strip()
        self.client = OpenAI(api_key=api_key, organization=organization)
        self.parallel_size = parallel_size
        self.max_retries = max_retries
        self.initial_wait_time = initial_wait_time
        self.end_wait_time = end_wait_time

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
        if prompt[-1]["role"] != "user":
            raise ValueError("Last message should be user")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs is not None,
            top_logprobs=logprobs,
            n=n,
            top_p=top_p,
        )
        response = response.model_dump()
        choices = response["choices"]
        for i in range(len(choices)):
            choices[i]["text"] = choices[i]["message"]["content"]
            if "logprobs" in choices[i]:
                choices[i]["raw_logprobs"] = choices[i]["logprobs"]
                choices[i]["tokens"] = [
                    x["token"] for x in choices[i]["logprobs"]["content"]
                ]
                _logprobs = []
                for x in choices[i]["logprobs"]["content"]:
                    token_logprobs = {
                        y["token"]: y["logprob"] for y in x["top_logprobs"]
                    }
                    token_logprobs[x["token"]] = x["logprob"]
                    _logprobs.append(token_logprobs)
                choices[i]["logprobs"] = _logprobs
            else:
                choices[i]["logprobs"] = None
                choices[i]["tokens"] = None
        return choices
    

@register_model("gpt-proxy")
class GPTProxy(BaseLLMAPI):
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
            end_wait_time (int, optional): End wait time. Defaults to 0.
        """
        from openai_client import OpenAIClient

        self.model_name = model_pt
        with open(key_path, "r") as f:
            api_key = f.read().strip()
        with open(account_path, "r") as f:
            user = f.read().strip()
        self.client = OpenAIClient(user=user, key=api_key)
        self.parallel_size = parallel_size
        self.max_retries = max_retries
        self.initial_wait_time = initial_wait_time
        self.end_wait_time = end_wait_time

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
        if prompt[-1]["role"] != "user":
            raise ValueError("Last message should be user")
        response = self.client.chat_completions_create(
            model=self.model_name,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs is not None,
            top_logprobs=logprobs,
            n=n,
            top_p=top_p,
        )
        choices = response["choices"]
        for i in range(len(choices)):
            choices[i]["text"] = choices[i]["message"]["content"]
            if "logprobs" in choices[i]:
                choices[i]["raw_logprobs"] = choices[i]["logprobs"]
                choices[i]["tokens"] = [
                    x["token"] for x in choices[i]["logprobs"]["content"]
                ]
                _logprobs = []
                for x in choices[i]["logprobs"]["content"]:
                    token_logprobs = {
                        y["token"]: y["logprob"] for y in x["top_logprobs"]
                    }
                    token_logprobs[x["token"]] = x["logprob"]
                    _logprobs.append(token_logprobs)
                choices[i]["logprobs"] = _logprobs
            else:
                choices[i]["logprobs"] = None
                choices[i]["tokens"] = None
        return choices


@register_model("o1")
class O1(BaseLLMAPI):
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
            end_wait_time (int, optional): End wait time. Defaults to 0.
        """
        from openai import OpenAI

        self.model_name = model_pt
        with open(key_path, "r") as f:
            api_key = f.read().strip()
        with open(account_path, "r") as f:
            organization = f.read().strip()
        self.client = OpenAI(api_key=api_key, organization=organization)
        self.parallel_size = parallel_size
        self.max_retries = max_retries
        self.initial_wait_time = initial_wait_time
        self.end_wait_time = end_wait_time

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
        if prompt[-1]["role"] != "user":
            raise ValueError("Last message should be user")
        if prompt[0]["role"] == "system":
            prompt[1]["content"] = prompt[0]["content"].strip() + "\n" + prompt[1]["content"]
            prompt = prompt[1:]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            logprobs=logprobs is not None,
            top_logprobs=logprobs,
            n=n,
            top_p=top_p,
        )
        response = response.model_dump()
        choices = response["choices"]
        for i in range(len(choices)):
            choices[i]["text"] = choices[i]["message"]["content"]
            choices[i]["logprobs"] = None
            choices[i]["tokens"] = None
        choices[0]["usage"] = response["usage"]
        return choices