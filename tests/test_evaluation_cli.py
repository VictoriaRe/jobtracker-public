import sys
import unittest
from unittest.mock import patch

import app


class EvaluationIsolationTests(unittest.TestCase):
    @patch("app.sheets_service")
    @patch("app.evaluation_cli")
    @patch("app.load_dotenv")
    @patch.object(sys, "argv", ["app.py", "eval", "run", "--annotations", "a.csv"])
    def test_evaluation_dispatch_never_connects_to_sheets(
        self, load_dotenv, evaluation_cli, sheets_service
    ):
        app.main()

        evaluation_cli.assert_called_once_with(["run", "--annotations", "a.csv"])
        sheets_service.assert_not_called()


if __name__ == "__main__":
    unittest.main()
