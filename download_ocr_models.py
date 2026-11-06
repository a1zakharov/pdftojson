#!/usr/bin/env python3
"""Download official Russian/English Tesseract models once, before offline use."""

import argparse
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parent / '.tessdata')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for language in ('rus', 'eng'):
        target = args.output / f'{language}.traineddata'
        if target.is_file() and target.stat().st_size > 1_000_000:
            print(f'Уже есть: {target}')
            continue
        url = ('https://raw.githubusercontent.com/tesseract-ocr/'
               f'tessdata_fast/main/{language}.traineddata')
        with urlopen(url, timeout=60) as response:
            data = response.read()
        if len(data) < 1_000_000:
            raise RuntimeError(f'Некорректный файл OCR-модели: {language}')
        temporary = target.with_suffix('.download')
        temporary.write_bytes(data)
        temporary.replace(target)
        print(f'Сохранено: {target}')


if __name__ == '__main__':
    main()
