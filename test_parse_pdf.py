import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pymupdf

from parse_pdf import (OCR, ROOT, clean_field_text, parse, read_checkbox,
                       select_template, usable_text)


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


class TextTests(unittest.TestCase):
    def test_form_underline_is_not_part_of_the_value(self):
        self.assertEqual(clean_field_text('________________\n01.02.2026'), '01.02.2026')
        self.assertEqual(clean_field_text('________\n123-456\n'), '123-456')

    def test_broken_character_map_requires_ocr(self):
        self.assertFalse(usable_text('\u0010=:5B0 On-prem'))
        self.assertTrue(usable_text('Анкета On-prem\nНазвание компании'))


class NoOCR:
    def read(self, image, psm=6):
        raise AssertionError('This generated PDF should be readable without OCR')


class LayoutTests(unittest.TestCase):
    def make_form(self, directory, template_name, title):
        template_path = ROOT / 'templates' / template_name
        template = json.loads(template_path.read_text(encoding='utf-8'))
        source = Path(directory) / 'unrelated-filename.pdf'
        # A generated blank form exercises the entire parser without committing
        # real questionnaires or customer contact details to the repository.
        with pymupdf.open() as document:
            for _ in range(template['page_count']):
                document.new_page(width=1548, height=2200)
            document[0].insert_text((120, 350), title, fontsize=30)
            for section in template['sections']:
                page = document[section['page'] - 1]
                for field in section['fields']:
                    if field['type'] == 'checkboxes':
                        for i, region in enumerate(field['regions']):
                            x0, y0, x1, y1 = region
                            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                            page.draw_rect(pymupdf.Rect(cx - 10, cy - 10, cx + 10, cy + 10),
                                           color=(0, 0, 0), width=1.5)
                            if field['id'] == 'windows_versions' and i == 1:
                                page.draw_circle(pymupdf.Point(cx, cy), 7,
                                                 color=(0, 0, 0), fill=(0, 0, 0))
                    elif field['id'] == 'completion_date':
                        page.insert_text((1160, 802), '01.02.2026', fontsize=30)
            document.save(source)
        return source

    def test_four_page_form_is_selected_and_parsed(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.make_form(folder, 'onprem.json', 'On-prem')
            actual = parse(source, None, NoOCR(), include_page_text=False)
        self.assertEqual(actual['page_count'], 4)
        self.assertEqual(actual['extraction']['template'], 'onprem.json')
        self.assertEqual(actual['extraction']['review'], [])
        values = {f['id']: f['value'] for s in actual['sections'] for f in s['fields']}
        self.assertEqual(len(values), 42)
        self.assertEqual(values['windows_versions'], ['10'])
        self.assertEqual(values['completion_date'], '01.02.2026')

    def test_three_page_form_still_works(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.make_form(folder, 'cloud.json', 'Cloud')
            actual = parse(source, None, NoOCR(), include_page_text=False)
        self.assertEqual(actual['page_count'], 3)
        self.assertEqual(actual['extraction']['template'], 'cloud.json')
        self.assertEqual(actual['extraction']['review'], [])

    def test_page_count_alone_does_not_select_a_template(self):
        class UnknownTitle:
            def read(self, image, psm=6):
                return 'A different questionnaire'

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'unknown.pdf'
            with pymupdf.open() as document:
                for _ in range(4):
                    document.new_page(width=1548, height=2200)
                document.save(source)
            with self.assertRaisesRegex(ValueError, 'Не найден шаблон'):
                select_template(source, UnknownTitle())

    def test_scanned_title_uses_ocr(self):
        class ScannedTitle:
            def read(self, image, psm=6):
                return 'Анкета On-prem'

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'scan.pdf'
            with pymupdf.open() as document:
                for _ in range(4):
                    document.new_page(width=1548, height=2200)
                document.save(source)
            selected = select_template(source, ScannedTitle())
        self.assertEqual(selected.name, 'onprem.json')


class QuestionnaireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = ROOT / 'Техническая Анкета по продукту ID_ облако (5) (1).pdf'
        cls.reference = cls.source.with_suffix('.json')
        if not cls.source.exists() or not cls.reference.exists():
            raise unittest.SkipTest('Для интеграционных проверок нужны исходные PDF и JSON')
        cls.ocr = OCR(tessdata=ROOT / '.tessdata')
        cls.template = ROOT / 'templates/cloud.json'

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
