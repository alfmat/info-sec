import unittest

from task import vigenere_encrypt_message

class TestVigenereEncryptMessage(unittest.TestCase):

    def test_1(self):
        c_expected = 'WDCNMFTKICEKZQGKVIAGQYXSGKTVRPRRGPCERXG'

        c = vigenere_encrypt_message('Imagination is more important than knowledge', 'ORCHESTRA')
        self.assertEqual(c, c_expected)


if __name__ == '__main__':
    unittest.main()
