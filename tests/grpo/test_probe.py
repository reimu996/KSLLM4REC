import ast
import inspect
import unittest

from ksllm4rec_grpo import probe


class ProbeGenerationCompatibilityTest(unittest.TestCase):
    def test_beam_generation_does_not_pass_removed_low_memory_flag(self) -> None:
        tree = ast.parse(inspect.getsource(probe))
        generate_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "generate"
        ]

        self.assertEqual(len(generate_calls), 1)
        keyword_names = {keyword.arg for keyword in generate_calls[0].keywords}
        self.assertNotIn("low_memory", keyword_names)


if __name__ == "__main__":
    unittest.main()
