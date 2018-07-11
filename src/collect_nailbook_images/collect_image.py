import os
import re
from time import sleep

from bs4 import BeautifulSoup
from requests.api import request

from .crawler_config import CrawlerConfig


def main():
    config = CrawlerConfig()
    start_crawling(config)


def start_crawling(config: CrawlerConfig):
    target_img = re.compile(r"https://cnv.nailbook.jp/photo/([0-9]+)/200")
    for page in range(1, config.max_page):
        print("start crawling page=%s" % page)
        try:
            ret = request("GET", config.url_template % dict(page=page))
        except Exception as e:
            print("error: page=%s, %s" % (page, e))
            continue

        bs = BeautifulSoup(ret.content, "html.parser")
        image_dir = "%s/%04d" % (config.image_dir, page)

        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        for img in bs.find_all("img"):
            try:
                img_src = img.attrs['src']
                match = target_img.search(img_src)
                if match:
                    image_id = match.group(1)
                    image_data = request("GET", img_src).content
                    with open("%s/%s.jpg" % (image_dir, image_id), "wb") as f:
                        f.write(image_data)
                    print("fetch %s" % image_id)
            except Exception as e:
                print(e)
        sleep(5)


if __name__ == '__main__':
    main()
