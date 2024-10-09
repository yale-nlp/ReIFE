from ..base_llm import BaseLLMAPI
from .registry import register_model


@register_model("gemini")
class GEMINI(BaseLLMAPI):
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

        self.model_name = model_pt
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        self.genai = genai
        self.HarmCategory = HarmCategory
        self.HarmBlockThreshold = HarmBlockThreshold
        with open(key_path, "r") as f:
            self.genai.configure(api_key=f.read().strip())
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
        model = self.genai.GenerativeModel(self.model_name)
        if n > 1:
            print(
                "Warning: GEMINI does not support multiple generations per prompt. Using n=1."
            )
        if logprobs is not None:
            print(
                "Warning: GEMINI does not support log probabilities. Ignoring logprobs."
            )
        if prompt[-1]["role"] != "user":
            raise ValueError("Last message should be user")
        if len([x for x in prompt if x["role"] == "user"]) > 1:
            raise ValueError("Only single round of conversation is supported")
        prompt = prompt[-1]["content"]
        response = model.generate_content(
            prompt,
            generation_config=self.genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                candidate_count=1,
                top_p=top_p,
            ),
            safety_settings={
                self.HarmCategory.HARM_CATEGORY_HATE_SPEECH: self.HarmBlockThreshold.BLOCK_NONE,
                self.HarmCategory.HARM_CATEGORY_HARASSMENT: self.HarmBlockThreshold.BLOCK_NONE,
                self.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: self.HarmBlockThreshold.BLOCK_NONE,
                self.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: self.HarmBlockThreshold.BLOCK_NONE,
            },
        )
        response_text = "No response"
        try:
            response_text = str(response.text)
        except Exception as e:
            print(e)
        return [{"text": response_text}]
