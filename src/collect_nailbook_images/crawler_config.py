import os


class CrawlerConfig:
    def __init__(self):
        self.image_dir = os.path.dirname(__file__) + '/../../data/image'
        self.url_template = 'https://nailbook.jp/design/?kind=popularity&page=%(page)s&part=100&treatment_case=4'
        self.max_page = 1000

