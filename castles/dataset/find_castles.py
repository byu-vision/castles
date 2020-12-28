'''Search for castles on Google maps.
'''

import time

# external dependencies
# pip install selenium
# also need to download chromedriver
import selenium
from selenium.webdriver import Chrome
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import (
    WebDriverException, TimeoutException, StaleElementReferenceException,
    ElementClickInterceptedException, ElementNotInteractableException)


class LocationScraper(object):
    def __init__(self):
        pass

    def launch(self, loc='https://www.google.com/maps'):
        '''Start up the webdriver and save a handle'''
        wd = Chrome()
        self.wd = wd
        if loc: wd.get(loc)
        return wd

    @staticmethod
    def wait_attr_change(el, attr='aria-label', initial=None, timeout=5):
        '''Wait until the attribute `attr` of web element `el` changes values'''
        f = 0.05
        if initial is None: initial = el.get_attribute(attr)
        start = time.time()
        for i in range(int(timeout/f)):
            s = time.time()
            au = el.get_attribute(attr)
            if au != initial: break
            else: time.sleep(max(0, f-(time.time()-s)))
        elapse = time.time() - start
        if au == initial:
            raise Exception(f'No change after {elapse:.2f} seconds')
        return elapse

    @staticmethod
    def wait_element(el, query_type, query, timeout=5):
        '''Wait for a specific element to appear on the page'''
        f = 0.05
        qfunc = 'find_element_by_' + query_type
        for i in range(int(timeout/f)):
            s = time.time()
            try:
                x = getattr(el, qfunc)(query)
                return x
            except WebDriverException as e:
                time.sleep(max(0, f-(time.time()-s)))
        raise TimeoutException('Unable to locate element in time')

    def get_result_list(self, pages=None):
        '''Return a list of all the results that match the query from
        `self.search_country`'''
        wd = self.wd
        def get_results_no_ads():
            results = wd.find_elements_by_class_name('section-result')
            results = [x for x in results if not x.get_attribute('data-result-ad-type')]
            return results

        def data_dict(r, search_pairs):
            f = r.find_element_by_class_name
            return {k: f(s).text for k,s in search_pairs}

        sr = 'section-result-'
        keys = ['details', 'location', 'description']
        search_pairs = [(k, f'{sr}{k}') for k in keys]

        self.wait_element(wd, 'class_name', 'section-result')
        results = get_results_no_ads()
        btnnext = self.wait_element(wd, 'css_selector', 'button[aria-label=" Next page "')

        page = 1
        data = []
        while True:
            for r in results:
                try:
                    d = dict(name=r.find_element_by_tag_name('h3').text,
                              **data_dict(r, search_pairs))
                    data.append(d)
                except StaleElementReferenceException:
                    pass
            exhausted = 'disabled' in btnnext.get_attribute('class')
            cutoff = pages and (page > pages)
            if exhausted or cutoff:
                break
            else:
                page += 1
                try:
                    initial = results[0].get_attribute('ved')
                    btnnext.click()
                    self.wait_attr_change(results[0], initial=initial)
                except StaleElementReferenceException: break
                except ElementClickInterceptedException: break
                except ElementNotInteractableException: break
                time.sleep(0.1)
                results = get_results_no_ads()
        return data

    def search_country(self, keyword, country):
        '''Initiate a search on Google maps of the form
        "{keyword} in {country}"'''
        wd = self.wd
        search = wd.find_element_by_id('searchboxinput')
        if not search.is_displayed():
            can = wd.find_element_by_tag_name('canvas')
            can.click()
            time.sleep(0.2)
            can.send_keys('+')
            time.sleep(0.2)
        search.send_keys(Keys.CONTROL+'a'+Keys.DELETE)
        time.sleep(0.2)
        search.send_keys(f'{keyword} in {country}\n')
