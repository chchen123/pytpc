import unittest
import pytpc.runtables as rt


class TestSanitize(unittest.TestCase):
    def test_case_change(self):
        s = 'CamelCaseStringWITHCAPS'
        e = 'camelcasestringwithcaps'
        r = rt._sanitize(s)
        self.assertEqual(r, e)

    def test_remove_spaces(self):
        s = 'string with spaces  two'
        e = 'string_with_spaces__two'
        r = rt._sanitize(s)
        self.assertEqual(r, e)

    def test_drop_parens(self):
        s = 'string with (units)'
        e = 'string_with'
        r = rt._sanitize(s)
        self.assertEqual(r, e)

    def test_everything(self):
        s = 'He:CO2 Pressure (torr)'
        e = 'heco2_pressure'
        r = rt._sanitize(s)
        self.assertEqual(r, e)


class TestParse(unittest.TestCase):
    def setUp(self):
        self.header = "a,b,c"
        self.header_list = ['a', 'b', 'c']
        self.data = ['{},{},{}'.format(a, a**2, a*5) for a in range(20)]
        self.csvdata = [self.header] + self.data

    def test_headers(self):
        r = rt.parse(self.csvdata)
        r_headers = list(list(r.values())[0].keys())
        for h in self.header_list:
            self.assertIn(h, r_headers)

if __name__ == '__main__':
    unittest.main()
