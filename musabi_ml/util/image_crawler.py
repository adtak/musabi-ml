import os

from pathlib import Path
from icrawler.builtin import BingImageCrawler
from typing import List


class ImageCrawler(object):
    def __init__(self, output_dir_path: Path) -> List[str]:
        self.output_dir = output_dir_path
        self.crawler = BingImageCrawler(storage={"root_dir": output_dir_path})

    def run(self, keyword: str, max_num: int) -> List[str]:
        filters = dict(
            size="large",
            type="photo",
            # license="creativecommons",
            layout="square")

        self.crawler.crawl(keyword=keyword, filters=filters, max_num=max_num)

        return self.get_results()

    def get_results(self) -> List[str]:
        return os.listdir(self.output_dir)
