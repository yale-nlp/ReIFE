import unittest
from ReIFE.models import get_model
import os
import logging

logging.basicConfig(level=logging.INFO)


class TestAPI(unittest.TestCase):
    def test_gemini(self):
        try:
            model_cls = get_model("gemini")
        except ValueError:
            self.fail("Failed to get gemini model")
        home_dir = os.getenv("HOME")
        model = model_cls(
            model_pt="gemini-1.0-pro",
            parallel_size=1,
            initial_wait_time=1,
            end_wait_time=1,
            max_retries=2,
            key_path=os.path.join(home_dir, "ReIFE/keys/google.key"),
            account_path=None,  # not needed
        )
        message = [{"role": "user", "content": "Introduce yourself in one sentence."}]
        response = model.get_response(
            prompt=message,
            n=1,
            max_tokens=128,
            temperature=1.0,
            top_p=1.0,
            logprobs=None,
        )
        self.assertIsInstance(response, list)
        self.assertIsInstance(response[0], dict)
        self.assertIn("text", response[0])
        logging.info(f"response: {response[0]['text']}")

    def test_gpt(self):
        try:
            model_cls = get_model("gpt")
        except ValueError:
            self.fail("Failed to get gpt model")
        home_dir = os.getenv("HOME")
        model = model_cls(
            model_pt="gpt-3.5-turbo-0125",
            parallel_size=1,
            initial_wait_time=1,
            end_wait_time=1,
            max_retries=2,
            key_path=os.path.join(home_dir, "ReIFE/keys/openai.key"),
            account_path=os.path.join(home_dir, "ReIFE/keys/openai.org"),
        )
        message = [{"role": "user", "content": "Introduce yourself in one sentence."}]
        response = model.get_response(
            prompt=message,
            n=2,
            max_tokens=128,
            temperature=1.0,
            top_p=1.0,
            logprobs=5,
        )
        self.assertIsInstance(response, list)
        self.assertIsInstance(response[0], dict)
        self.assertEqual(len(response), 2)
        self.assertIn("text", response[0])
        self.assertIn("logprobs", response[0])
        self.assertIn("tokens", response[0])
        logging.info(f"response: {response[0]['text']}")

    def test_cohere_chat(self):
        try:
            model_cls = get_model("cohere_chat")
        except ValueError:
            self.fail("Failed to get cohere_chat model")
        home_dir = os.getenv("HOME")
        model = model_cls(
            model_pt="command-r-plus",
            parallel_size=1,
            initial_wait_time=1,
            end_wait_time=1,
            max_retries=2,
            key_path=os.path.join(home_dir, "ReIFE/keys/cohere.key"),
            account_path=None,  # not needed
        )
        message = [{"role": "user", "content": "Introduce yourself in one sentence."}]
        response = model.get_response(
            prompt=message,
            n=1,
            max_tokens=128,
            temperature=1.0,
            top_p=1.0,
            logprobs=None,
        )
        self.assertIsInstance(response, list)
        self.assertIsInstance(response[0], dict)
        self.assertIn("text", response[0])
        logging.info(f"response: {response[0]['text']}")


if __name__ == "__main__":
    unittest.main()
