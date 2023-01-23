
# Generate unit tests for the class ASTHandler
import unittest

from asthandler import ASTHandler

class TestASTHandler(unittest.TestCase):
    def setUp(self):
        self.asthandler = ASTHandler()
        self.prefix = ['+', 'num1', '*', 'num2', 'num3']
        self.suffix = ['num1', 'num2', 'num3', '*', '+']
        self.infix = ['(', 'num1', '+', '(', 'num2', '*', 'num3', ')', ')']

    def test_prefix2suffix(self):
        self.assertEqual(self.asthandler.prefix2suffix(self.prefix), " ".join(self.suffix))

    def test_prefix2infix(self):
        self.assertEqual(self.asthandler.prefix2infix(self.prefix), " ".join(self.infix))

    def test_suffix2prefix(self):
        self.assertEqual(self.asthandler.suffix2prefix(self.suffix), " ".join(self.prefix))

    def test_suffix2infix(self):
        self.assertEqual(self.asthandler.suffix2infix(self.suffix), " ".join(self.infix))

    def test_infix2prefix(self):
        self.assertEqual(self.asthandler.infix2prefix(self.infix), " ".join(self.prefix))

    def test_infix2suffix(self):
        self.assertEqual(self.asthandler.infix2suffix(self.infix), " ".join(self.suffix))