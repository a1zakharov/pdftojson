import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pymupdf

from parse_pdf import OCR, ROOT, parse, read_checkbox


class CheckboxTests(unittest.TestCase):
    def image(self):
        image = np.full((40, 40), 255, np.uint8)
        cv2.rectangle(image, (9, 9), (30, 30), 0, 2)
        return image

    def test_blank_box(self):
        self.assertIs(read_checkbox(self.image(), [0, 0, 40, 40])['checked'], False)

    def test_dot(self):
        image = self.image()
        cv2.circle(image, (20, 20), 7, 0, -1)
        self.assertIs(read_checkbox(image, [0, 0, 40, 40])['checked'], True)

    def test_cross(self):
        image = self.image()
        cv2.line(image, (13, 13), (26, 26), 0, 3)
        cv2.line(image, (13, 26), (26, 13), 0, 3)
        self.assertIs(read_checkbox(image, [0, 0, 40, 40])['checked'], True)

    def test_missing_box_is_not_an_unchecked_answer(self):
        image = np.full((40, 40), 255, np.uint8)
        reading = read_checkbox(image, [0, 0, 40, 40])
        self.assertIsNone(reading['checked'])
        self.assertEqual(reading['issue'], 'checkbox_not_found')

    def test_faint_mark_requires_review(self):
        image = self.image()
        image[18:20, 15:25] = 0
        reading = read_checkbox(image, [0, 0, 40, 40])
        self.assertIsNone(reading['checked'])
        self.assertEqual(reading['issue'], 'ambiguous_checkbox')


class QuestionnaireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = ROOT / 'Техническая Анкета по продукту ID_ облако (5) (1).pdf'
        cls.reference = cls.source.with_suffix('.json')
        if not cls.source.exists() or not cls.reference.exists():
            raise unittest.SkipTest('Для интеграционных проверок нужны исходные PDF и JSON')
        cls.ocr = OCR(tessdata=ROOT / '.tessdata')
        cls.template = ROOT / 'templates/id_cloud.json'

    @staticmethod
    def values(data):
        return {field['id']: field['value']
                for section in data['sections'] for field in section['fields']}

    def test_all_30_values_match_visual_reference(self):
        expected = json.loads(self.reference.read_text(encoding='utf-8'))
        actual = parse(self.source, self.template, self.ocr, include_page_text=False)
        self.assertEqual(len(self.values(actual)), 30)
        self.assertEqual(self.values(actual), self.values(expected))
        self.assertEqual(actual['extraction']['review'], [])

    def test_changed_checkbox_changes_the_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            modified = Path(folder) / 'changed.pdf'
            with pymupdf.open(self.source) as document:
                page = document[0]
                sx, sy = page.rect.width / 1548, page.rect.height / 2200
                clear = pymupdf.Rect(125 * sx, 1264 * sy, 143 * sx, 1282 * sy)
                page.draw_rect(clear, color=None, fill=(1, 1, 1), overlay=True)
                page.draw_circle(pymupdf.Point(655 * sx, 1273 * sy), 7 * sx,
                                 color=(0, 0, 0), fill=(0, 0, 0), overlay=True)
                document.save(modified)
            actual = parse(modified, self.template, self.ocr, include_page_text=False)
        self.assertEqual(self.values(actual)['ldap_protocol'], ['LDAPS'])
        self.assertEqual(actual['extraction']['review'], [])

    def test_wrong_page_count_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            wrong = Path(folder) / 'wrong.pdf'
            with pymupdf.open() as document:
                document.new_page()
                document.save(wrong)
            with self.assertRaisesRegex(ValueError, 'Ожидается страниц'):
                parse(wrong, self.template, self.ocr, include_page_text=False)


if __name__ == '__main__':
    unittest.main()
