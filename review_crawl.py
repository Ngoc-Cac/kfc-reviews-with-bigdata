import base64
import configparser
import os.path as osp

from pathlib import Path
from traceback import format_exc


from selenium.webdriver import (
    Firefox,
    FirefoxOptions
)
from selenium.webdriver.common.by import By

from utils import (
    get_place_meta,
    get_review_ratings,
    get_review_texts,
    scroll_reviews
)

# logging stuff
import log_setup # this just sets up logging, it's not meant to be used as a module
import logging
logger = logging.getLogger(__name__)


def crawl_process() -> tuple[list[str], list[str]]:
    # go to reviews tab by clicking reviews button
    driver.find_element(By.XPATH, "//button[@class='hh2c6 ' and contains(., 'Bài đánh giá')]")\
          .click()

    logger.info('Loading reviews...')
    scroll_reviews(driver, NUM_REVIEWS, LOADING_TIMEOUT)

    logger.info('Crawling reviews data...')
    try:
        reviews = get_review_texts(driver)
        ratings = get_review_ratings(driver)
    except Exception as e:
        logger.critical(f"""Exception occured while trying to crawl reviews for {url}
{format_exc(e)}""")
        return
    return reviews, ratings

def write_to_file(place_id: int, reviews: list[str], ratings: list[str]):
    # write to csv, strings are encoded as base 64 under utf-8 encoding
    for review, rating in zip(reviews, ratings):
        b64_encoded_review = base64.b64encode(bytes(review, encoding='utf-8'))
        # b64encode returns a bytes object, we decode it into a utf-8 encoded string
        b64_encoded_review = b64_encoded_review.decode()
        reviews_file.write(f'{b64_encoded_review},{rating},{place_id}\n')




parser = configparser.ConfigParser()
print('Getting things ready before crawling...')
try:
    parser.read('crawler.conf')
    LOADING_TIMEOUT = float(parser['crawler_options']['loading_timeout'])
    HEADLESS = bool(parser['firefox_options']['headless'])
    NUM_REVIEWS = int(parser['crawler_options']['max_reviews'])
except Exception as e:
    logger.critical(f'Could not parse crawler.conf!\n{format_exc(e)}')

# get links and create output file
with open(parser['links']['file_location'], encoding='utf-8') as file:
    links = file.read().split('\n')
if not osp.exists('output/'):
    Path.mkdir('output')
reviews_file = open('output/reviews.csv', 'w', encoding='utf-8')
metedata_file = open('output/place_metadata.csv', 'w', encoding='utf-8')
reviews_file.write('review_b64encoded,rating,place_id\n')
metedata_file.write('id,url,address,price_range\n')



options = FirefoxOptions()
options.set_preference("intl.accept_languages", parser['firefox_options']['language'])
if HEADLESS: options.add_argument('--headless')

logger.info('Initializing driver...')
driver = Firefox(options=options)
logger.info('Driver initialized...')


META_EXC_PLACEHOLDER = {'address': '', 'price_range': ''}
for i, url in enumerate(links):
    print(f'Now crawling from {url}')
    logger.info(f'Connecting to {url}')
    driver.get(url)


    print('Getting place overview...')
    try:
        meta_data = get_place_meta(driver)
    except Exception as e:
        logger.critical(f"""Exception occured while trying to get place overview for url {url}!
{format_exc(e)}""")
        meta_data = META_EXC_PLACEHOLDER
    metedata_file.write(f'{i},"{url}","{meta_data["address"]}",{meta_data["price_range"]}\n')


    results = crawl_process()
    if results is None:
        print('Crawling unsuccesful! Check logs for details...')
    else:
        print('Crawling finished! Saving to file...')
        write_to_file(i, *results)
        print('Saved to file!')


driver.close()
metedata_file.close()
reviews_file.close()