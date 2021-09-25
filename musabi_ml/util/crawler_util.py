import os
import urllib
from pathlib import Path
from typing import List

from icrawler.builtin import BingImageCrawler


class ImageCrawler(object):
    def __init__(self, output_dir_path: Path) -> List[str]:
        self.output_dir = output_dir_path
        self.crawler = BingImageCrawler(storage={"root_dir": output_dir_path})

    def run(self, keyword: str, max_num: int) -> List[str]:
        filters = dict(
            size="large",
            type="photo",
            layout="square")

        self.crawler.crawl(keyword=keyword, filters=filters, max_num=max_num)

        return self.get_results()

    def get_results(self) -> List[str]:
        return os.listdir(self.output_dir)


def download_file(url: str, download_file_path: Path):
    try:
        with urllib.request.urlopen(url) as web_file, open(download_file_path, 'wb') as local_file:
            local_file.write(web_file.read())
    except urllib.error.URLError as e:
        print(e)
