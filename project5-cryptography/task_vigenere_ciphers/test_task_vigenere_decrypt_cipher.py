import unittest

from task import vigenere_decrypt_cipher

class TestVigenereDecryptCipher(unittest.TestCase):

    def test_1(self):
        m_expected = 'THEREARENOSUCHTHINGSASAPPLIEDSCIENCESONLYAPPLICATIONSOFSCIENCE'

        m = vigenere_decrypt_cipher('BRYMEUFBEGAEWCTBWKXKICUKPFWBUKKSYICYGLEDGKJKLCQXKAWXMJFMQFVFKO', 'IKUVAUOXRS')
        self.assertEqual(m, m_expected)


if __name__ == '__main__':
    unittest.main()
