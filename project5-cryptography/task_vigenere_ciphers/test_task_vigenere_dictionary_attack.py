import unittest

from task import vigenere_dictionary_attack

class TestVigenereDictionaryAttack(unittest.TestCase):

    def test_1(self):
        m_expected = 'PHOTOGRAPHYSEPARATELYPROTECTEDLITIGATIONTREATMENTCONSIDERINGANNOUNCEMENTIDENTICALLIABILITIESINSPECTOR'

        m = vigenere_dictionary_attack('XKSGHOWITYGVICTZFBICGSVBMMHBIUTLXVZIYQSEBUINMUJVXTWQWVWMWQRXIQRBNVHMQVVWMQXVYQGRTOMNUQQQXZMVMALXJKXFZ')
        self.assertEqual(m, m_expected)


if __name__ == '__main__':
    unittest.main()
