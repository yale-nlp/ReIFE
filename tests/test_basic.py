import unittest

class TestPackageImports(unittest.TestCase):
    def test_import_package(self):
        try:
            import ReIFE
        except ImportError:
            self.fail("Failed to import ReIFE")

    def test_import_base_llm(self):
        try:
            from ReIFE.base_llm import BaseLLM, BaseVLLM, BaseLLMAPI
        except ImportError:
            self.fail("Failed to import BaseLLM, BaseVLLM, BaseLLMAPI")

    def test_import_methods(self):
        try:
            from ReIFE.methods import get_method, get_parser
        except ImportError:
            self.fail("Failed to import get_method, get_parser")

    def test_import_models(self):
        try:
            from ReIFE.models import get_model
        except ImportError:
            self.fail("Failed to import get_model")

    def test_import_evaluator(self):
        try:
            from ReIFE.evaluator import PairwiseEvaluator
        except ImportError:
            self.fail("Failed to import PairwiseEvaluator")


class TestCoreFunctionality(unittest.TestCase):
    def test_instantiate_evaluator(self):
        from ReIFE.evaluator import PairwiseEvaluator
        try:
            evaluator = PairwiseEvaluator()
        except Exception as e:
            self.fail(f"Failed to instantiate PairwiseEvaluator: {e}")

    

