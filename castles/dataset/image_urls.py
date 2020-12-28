'''Get image urls for specific castles from Google maps'''

import fcntl
import json
from pathlib import Path
import re
import socket
import time
import tqdm

# external dependencies
# pip install selenium
# also need to download chromedriver
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import selenium.common.exceptions as se


hostname = socket.gethostname()
OUTFILE = f'meta/maps-img-urls_{hostname}.txt'
COMPLETE = f'meta/maps-urls-scraped_{hostname}.txt'


def multi_wait(wd, eqs, wait=10):
    '''Wait until one of multiple possible elements is loaded.
    eqs: list containing pairs of (function_name, query_string) strings
         that search for an element on the page.
    '''
    interval = 0.05
    stop = False
    last_exception = None
    for _ in range(int(wait/interval)):
        st = time.time()
        for func, param in eqs:
            try:
                el = getattr(wd, func)(param)
                stop = True
                break
            except se.NoSuchElementException as exc:
                last_exception = exc
        if stop: return el
        else: time.sleep(max(0, time.time()-st))
    raise last_exception


def wait_condition(condition_func, *args, wait=10):
    interval = 0.05
    stop = False
    for _ in range(int(wait/interval)):
        st = time.time()
        result = condition_func(*args)
        if result: return result
        else: time.sleep(max(0, time.time()-st))
    return False


def get_wait(wd, q, by, time=10):
    return WebDriverWait(wd, time).until(
        EC.presence_of_element_located((by, q)))

def scroll_to(wd, elem):
    s = 'arguments[0].scrollIntoView({behavior:"auto",block:"center",inline:"center"});'
    wd.execute_script(s, elem)


def get_driver(headless=False):
    opts = Options()
    if headless:
        opts.add_argument('--headless')
    wd = webdriver.Chrome(options=opts)
    return wd


def attempt_scroll_click(wd, el, attempts=20):
    for i in range(attempts):
        try:
            scroll_to(wd, el)
            time.sleep(0.05)
            el.click()
            break
        except:
            pass
    

def get_urls(query, n=200, headless=False, wd=None):
    '''Get image urls from Google maps based on a query which should lead to a
    Google maps location: for example, "bodiam castle england".'''
    # there are several different response layouts that might occur based on
    # the query, which this function tries to handle. it's possible things
    # will change in the future, so no gaurentees on how long this will work
    wd_exists = wd is not None
    if not wd_exists: wd = get_driver(headless)
    if query.startswith('http'):
        wd.get(query)
    else:
        wd.get('https://www.google.com/maps')
        get_wait(wd, 'searchboxinput', By.ID).send_keys(query+'\n')

    prefix = 'find_element_by_'
    attempts = [
        (prefix+'class_name', 'section-carouselphoto-photo-container-shim'),
        (prefix+'xpath',"//div[@class='section-image-pack-image-container']/.."),
        (prefix+'css_selector','div.section-result[role="link"]')
    ]
    urls = []
    try:
        el = multi_wait(wd, attempts, 10)
        class_name = el.get_attribute('class')
        attempt_scroll_click(wd, el)

        if class_name == 'section-result':
            el = multi_wait(wd, attempts[:2], 10)
            attempt_scroll_click(wd, el)

        def get_more(wd, n):
            ims = wd.find_elements_by_class_name('gallery-image-high-res')
            if len(ims) > n: return ims
            else: return False
        img_loaded = (lambda el: 'loaded' in el.get_attribute('class'))

        imgs = []
        nn = 0
        get_wait(wd, 'gallery-image-high-res', By.CLASS_NAME)
        while True:
            ims = wait_condition(get_more, wd, nn, wait=4)
            if ims: imgs = ims
            else: break
            for img in imgs[nn:]:
                wd.execute_script('arguments[0].scrollIntoView();', img)
                wait_condition(img_loaded, img)
            if len(imgs) == nn or len(imgs) >= n:
                break
            nn = len(imgs)

        p = re.compile(r'url[(]"(.*?)"[)]')
        urls = [re.search(p, img.get_attribute('style')) for img in imgs]
        urls = [x.groups()[0] for x in urls if x]
    except se.WebDriverException:
        pass
    finally:
        if not wd_exists:
            wd.close()
    return urls


def from_data_list(data, start=0, end=None, name_col=1, country_col=2,
                   url_cols=[-6,-5], outfile=OUTFILE, completed_log=COMPLETE):
    '''Given a list of information about different castles, construct a query for
    each one and search Google maps for image urls. This can be run on different
    subsets of the data on different machines, which can optionally write to the
    same log files on a shared filesystem.'''
    completed = set()
    if Path(completed_log).is_file():
        completed = set(open(completed_log).read().strip().split('\n'))

    it = tqdm.tqdm(data[start:end])
    try:
        for i, row in enumerate(it):
            nimgs = 0
            for url in (row[url_idx] for url_idx in url_cols):
                if not url or url in completed: continue
                desc = '{} {}'.format(row[name_col], row[country_col])
                it.set_description_str(f'{desc[:30]:<30}')
                imgs = get_urls(url, 1000-nimgs, headless=True)
                nimgs += len(imgs)
                imtxt = '\n{}'.format('\n'.join(imgs)) if imgs else ''
                with open(outfile, 'a') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    f.write('{} -- {}\n{}{}\n\n'.format(
                        row[name_col], row[country_col], url, imtxt))
                    fcntl.flock(f, fcntl.LOCK_UN)
                with open(completed_log, 'a') as f:
                    fcntl.flock(f, fcntl.LOCK_UN)
                    f.write(f'{url}\n')
                    fcntl.flock(f, fcntl.LOCK_UN)
                completed.add(url)
    finally:
        it.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='Castle name or path to json file')
    parser.add_argument('dest', help='JSON file to save results')
    parser.add_argument('-n', type=int, default=200, help='Number of image urls to get')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    args = parser.parse_args()

    urls = get_urls(args.name, args.n, args.headless)
    print('\n'.join(urls))
    print(len(urls))
