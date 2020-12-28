'''Download Google maps user-uploaded images'''

import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import time
import urllib

# external dependencies
# pip install filetype pillow requests tqdm unihandecode
import filetype
from PIL import Image
import requests
import tqdm
from unihandecode import unidecode


def convert_name(name, country):
    '''Convert unicode name to ascii'''
    name = unidecode(name).lower().replace(' ', '-')
    country = unidecode(country).lower().replace(' ', '-')
    return f'{name}_{country}'


def get_hash(s, nbytes=8):
    '''Get a hash code for a string'''
    hasher = hashlib.shake_128()
    hasher.update(urllib.parse.unquote_to_bytes(s))
    return hasher.hexdigest(nbytes)


def download_images(name, country, urls, dest,
                    res=500, pause=0.2, print_=None):
    '''Download images of a specified location, from a list of image urls.
    
    Args:
        name (str): name of the castle
        country (str): name of the country where the castle is located
        urls (list[str]): list of image urls to download
        dest (str): root destination folder
        res (int): larger-side image resolution
        pause (float): time to pause between downloads
        print_ (callable or None): if None, use regular print. Otherwise,
            should be a callable that handles printing output
    '''
    def make_print():
        if print_ is None:
            def print_inner(s, **kwargs):
                print(s, **kwargs)
            return print_inner
        else:
            return print_
    print_ = make_print()
    cname = convert_name(name, country)
    dest = Path(dest)
    dest = dest.joinpath(cname)
    dest.mkdir(exist_ok=True)
    files = set(x.stem for x in dest.iterdir())
    #last = int(files[-1]) if len(files) else 0

    headers = {'User-Agent':
              ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
               '(KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36')}

    print_(f'Starting downloads: {cname[:30]} -- found {len(files)} pre-existing')

    files = set(files)
    counter = 0
    start_time = time.time()
    for i, url in enumerate(urls):
        if not url.startswith('http'): continue
        #imgname = f'{i:>04}_{urlhash}'
        imgname = get_hash(url)
        if imgname in files: continue
        if 'googleusercontent' in url:
            url = re.sub(r'=s[0-9]+-', f'=s{res}-', url)
        try:
            response = requests.get(url, headers=headers, timeout=5)
            content = response.content
            ext = filetype.guess_extension(content)
            with open(dest.joinpath(f'{imgname}.{ext}'),'wb') as imgfile:
                imgfile.write(content)
            #img = Image.open(BytesIO(content))
            #fmat = img.format
            #img.save(dest.joinpath(f'{imgname}.{fmat.lower()}'), format=fmat)
            counter += 1
            time.sleep(pause)
        except Exception as exc:
            print_(f'{name[:20]} ({i}): {str(exc)}')
            continue

    elapse = time.time() - start_time
    print_(f'{name[:20]}: successfully downloaded {counter} / {len(urls)} ({elapse:.2f} s)')


def download_all(urls, keys):
    it = tqdm.tqdm(keys)
    try:
        for k in it:
            download_images(k, urls[k], 'castles', pause=.2, print_=it.write)
    finally:
        it.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('urls', help='Path to json file with urls')
    parser.add_argument('-k', '--keys', default='',
                        help='File containing names to download')
    args = parser.parse_args()
    urls = json.load(open(args.urls))
    if args.keys:
        ks = json.load(open(args.keys))
    else:
        ks = list(urls.keys())
    download_all(urls, ks)

