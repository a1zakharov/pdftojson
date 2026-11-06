#!/usr/bin/env python3
"""Read a questionnaire locally using a layout template, OCR and pixels."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pymupdf


ROOT = Path(__file__).resolve().parent


def usable_text(text):
    """Reject broken PDF character maps rather than returning their control codes."""
    return bool(text.strip()) and not any(
        (ord(char) < 32 and char not in '\n\r\t') or char == '\ufffd'
        for char in text)


def clean_field_text(text):
    # Printed form underlines can overlap a value's extraction rectangle even
    # when the actual line sits below its visible crop.
    text = re.sub(r'_{3,}', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return '\n'.join(line.strip() for line in text.splitlines() if line.strip())


def pdf_region(page, region, reference_size):
    width, height = reference_size
    x0, y0, x1, y1 = region
    return pymupdf.Rect(x0 / width * page.rect.width, y0 / height * page.rect.height,
                        x1 / width * page.rect.width, y1 / height * page.rect.height)


def select_template(source, ocr, directory=None):
    """Match both page count and a title anchor; never infer a layout from count alone."""
    directory = Path(directory) if directory else ROOT / 'templates'
    matches = []
    with pymupdf.open(source) as document:
        images = {}
        recognized = {}
        for path in sorted(directory.glob('*.json')):
            template = json.loads(path.read_text(encoding='utf-8'))
            anchor = template.get('match')
            if template['page_count'] != document.page_count or not anchor:
                continue
            page_index = anchor['page'] - 1
            page = document[page_index]
            rect = pdf_region(page, anchor['region'], template['reference_size'])
            title = page.get_text('text', clip=rect)
            expected = anchor['text'].casefold()
            if expected not in title.casefold():
                cache_key = (page_index, tuple(template['reference_size']))
                if cache_key not in images:
                    images[cache_key] = render_page(page, template['reference_size'])
                region_key = (cache_key, tuple(anchor['region']))
                if region_key not in recognized:
                    recognized[region_key] = ocr.read(crop(images[cache_key], anchor['region']))
                title = recognized[region_key]
            if expected in title.casefold():
                matches.append(path)
        if not matches:
            raise ValueError(f'Не найден шаблон для документа ({document.page_count} стр.). '
                             'Укажите подходящий шаблон через --template; '
                             'для другого макета нужно создать новый шаблон.')
    if len(matches) > 1:
        raise ValueError('Подошло несколько шаблонов. Укажите --template: '
                         + ', '.join(path.name for path in matches))
    return matches[0]


def render_page(page, reference_size):
    width, height = reference_size
    if abs(page.rect.width / page.rect.height - width / height) > 0.005:
        raise ValueError("Пропорции страницы не соответствуют шаблону")
    pix = page.get_pixmap(matrix=pymupdf.Matrix(width / page.rect.width,
                                               height / page.rect.height),
                          colorspace=pymupdf.csGRAY, alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def crop(image, box):
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
        raise ValueError(f"Область за пределами страницы: {box}")
    return image[y0:y1, x0:x1]


def read_checkbox(image, box):
    """Validate a square outline, then measure ink inside it, excluding its border."""
    patch = crop(image, box)
    mask = np.uint8(patch < 160) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (0.4 * patch.shape[1] <= w <= 0.9 * patch.shape[1]
                and 0.4 * patch.shape[0] <= h <= 0.9 * patch.shape[0]
                and 0.85 <= w / h <= 1.15):
            continue
        if abs(x + w / 2 - patch.shape[1] / 2) > 5:
            continue
        if abs(y + h / 2 - patch.shape[0] / 2) > 5:
            continue
        # Letters and arbitrary filled blobs should not pass as checkbox outlines.
        if cv2.contourArea(contour) < 0.65 * w * h:
            continue
        candidates.append((x, y, w, h))
    if not candidates:
        return {'checked': None, 'ink_ratio': None, 'issue': 'checkbox_not_found'}
    x, y, w, h = max(candidates, key=lambda b: b[2] * b[3])
    interior = mask[y + round(h * .25):y + round(h * .75),
                    x + round(w * .25):x + round(w * .75)]
    ratio = float(np.mean(interior > 0))
    if ratio <= 0.08:
        checked, issue = False, None
    elif ratio >= 0.30:
        checked, issue = True, None
    else:
        checked, issue = None, 'ambiguous_checkbox'
    return {'checked': checked, 'ink_ratio': round(ratio, 4), 'issue': issue}


class OCR:
    def __init__(self, language='rus+eng', tessdata=None):
        self.language = language
        self.tessdata = tessdata
        self.validated = False

    def validate(self):
        if self.validated:
            return
        if not shutil.which('tesseract'):
            raise RuntimeError('Не найден tesseract. Установите Tesseract OCR.')
        args = ['tesseract']
        if self.tessdata:
            args += ['--tessdata-dir', str(self.tessdata)]
        result = subprocess.run(args + ['--list-langs'], capture_output=True,
                                text=True, check=True)
        installed = set(result.stdout.splitlines()[1:])
        missing = set(self.language.split('+')) - installed
        if missing:
            raise RuntimeError('Нет OCR-моделей: ' + ', '.join(sorted(missing))
                               + '. См. README.md или укажите --tessdata.')
        self.validated = True

    def read(self, image, psm=6):
        self.validate()
        # Padding helps Tesseract recognize tightly cropped single-line values.
        image = cv2.copyMakeBorder(image, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)
        ok, encoded = cv2.imencode('.png', image)
        if not ok:
            raise RuntimeError('Не удалось подготовить изображение для OCR')
        args = ['tesseract', 'stdin', 'stdout', '-l', self.language, '--psm', str(psm)]
        if self.tessdata:
            args += ['--tessdata-dir', str(self.tessdata)]
        result = subprocess.run(args, input=encoded.tobytes(), capture_output=True,
                                check=True)
        return result.stdout.decode('utf-8').strip()


def read_text_field(page, image, field, ocr):
    region = crop(image, field['region'])
    if float(np.mean(region < 160)) < 0.002:
        return None, 'blank', None
    rect = pdf_region(page, field['region'], (image.shape[1], image.shape[0]))
    text = clean_field_text(page.get_text('text', clip=rect))
    method = 'pdf_text'
    if not usable_text(text):
        text, method = ocr.read(region, field.get('psm', 7)), 'tesseract'
    text = clean_field_text(text)
    if not text:
        return None, method, 'unrecognized_text'
    if field.get('pattern') and not re.fullmatch(field['pattern'], text):
        return text, method, 'unexpected_text_format'
    return text, method, None


def parse(source, template_path, ocr, include_page_text=True):
    if template_path is None:
        template_path = select_template(source, ocr)
    template = json.loads(Path(template_path).read_text(encoding='utf-8'))
    result = {key: copy.deepcopy(template[key])
              for key in ('title', 'brand', 'product', 'language', 'sections', 'footer')}
    result.update(source_file=Path(source).name, page_count=template['page_count'],
                  value_conventions={
                      'text': 'Строка или null, если поле пустое.',
                      'checkboxes': 'Массив отмеченных вариантов или null, если отметок нет. '
                                    'При ошибке чтения null требует проверки review.',
                  },
                  extraction={'engine': 'PyMuPDF + OpenCV + Tesseract',
                              'template': Path(template_path).name,
                              'review': [], 'page_text': []})
    with pymupdf.open(source) as document:
        if document.page_count != template['page_count']:
            raise ValueError(f"Ожидается страниц: {template['page_count']}, "
                             f"получено: {document.page_count}")
        for index, page in enumerate(document):
            image = render_page(page, template['reference_size'])
            if include_page_text:
                text = page.get_text('text').strip()
                method = 'pdf_text'
                # Sparse layers may contain only a date and checkbox marks.
                if not usable_text(text) or sum(char.isalpha() for char in text) < 200:
                    text, method = ocr.read(image, psm=3), 'tesseract'
                result['extraction']['page_text'].append(
                    {'page': index + 1, 'method': method, 'text': text})
            for section in result['sections']:
                if section['page'] != index + 1:
                    continue
                for field in section['fields']:
                    issues = []
                    if field['type'] == 'checkboxes':
                        if len(field['options']) != len(field['regions']):
                            raise ValueError(f"Число областей не совпадает: {field['id']}")
                        details = []
                        for option, box in zip(field['options'], field.pop('regions')):
                            reading = read_checkbox(image, box)
                            details.append({'option': option, 'region': box, **reading})
                            if reading['issue']:
                                issues.append(f"{option}: {reading['issue']}")
                        selected = [item['option'] for item in details if item['checked']]
                        field['value'] = None if issues else selected or None
                        field['reading'] = {'method': 'checkbox_pixels', 'options': details}
                    else:
                        value, method, issue = read_text_field(page, image, field, ocr)
                        field['value'] = value
                        field['reading'] = {'method': method, 'region': field.pop('region')}
                        field.pop('psm', None)
                        field.pop('pattern', None)
                        if issue:
                            issues.append(issue)
                    if issues:
                        result['extraction']['review'].append(
                            {'page': index + 1, 'field': field['id'], 'issues': issues})
    result['extraction']['needs_review'] = bool(result['extraction']['review'])
    return result


def to_answers(data):
    """Export every field as a readable question/value pair without diagnostics."""
    counts = Counter(field['label'] for section in data['sections']
                     for field in section['fields'])
    answers = {}
    for section in data['sections']:
        for field in section['fields']:
            key = field['label']
            if counts[key] > 1:
                key = f"{key} ({section['title']})"
            if key in answers:
                raise ValueError(f'Повторяющееся название поля: {key}. '
                                 'Уточните названия в шаблоне.')
            value = field['value']
            if field['type'] == 'checkboxes' and value is not None:
                value = value[0] if len(value) == 1 else value or None
            answers[key] = value
    return answers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf', type=Path)
    parser.add_argument('-o', '--output', type=Path)
    parser.add_argument('--template', type=Path,
                        help='Путь к шаблону; по умолчанию определяется по заголовку и числу страниц')
    parser.add_argument('--lang', default='rus+eng')
    local_models = ROOT / '.tessdata'
    parser.add_argument('--tessdata', type=Path,
                        default=local_models if local_models.is_dir() else None)
    parser.add_argument('--detailed', action='store_true',
                        help='Сохранить подробный результат с координатами и диагностикой')
    parser.add_argument('--no-page-text', action='store_true',
                        help='Не включать полный текст страниц в режиме --detailed')
    args = parser.parse_args()
    output = args.output or args.pdf.with_suffix('.python.json')
    if output.resolve() == args.pdf.resolve():
        parser.error('Выходной файл не должен совпадать с исходным PDF')
    try:
        data = parse(args.pdf, args.template, OCR(args.lang, args.tessdata),
                     include_page_text=args.detailed and not args.no_page_text)
        exported = data if args.detailed else to_answers(data)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(exported, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f'Ошибка: {error}', file=sys.stderr)
        return 1
    print(f'JSON: {output}')
    if data['extraction']['needs_review']:
        labels = {field['id']: field['label'] for section in data['sections']
                  for field in section['fields']}
        print('Есть неоднозначные поля:', file=sys.stderr)
        for item in data['extraction']['review']:
            print(f"  Стр. {item['page']}: {labels[item['field']]}", file=sys.stderr)
        if args.detailed:
            print('Подробности: extraction.review.', file=sys.stderr)
        else:
            print('Для подробной диагностики используйте --detailed.', file=sys.stderr)
        return 2
    print('Все области прочитаны без диагностированных ошибок.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
