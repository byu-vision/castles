'''Download tarball of Castles dataset. This will download the dataset in 5
fragments and then reassemble them into a single .tgz file. This requires a
little over 100 GB of free disk space. Once the final castles.tgz file has been
assembled, the *.tgz_fragment* files can be deleted, leaving the ~50GB
castles.tgz file to be extracted.
'''
import hashlib
import json
import os
import requests
import subprocess
import sys
from tqdm import tqdm

METADATA = {
    "fragments":[
        {"boxfile":"edizwblz2jbv43wps9lduettt7b78a2y.tgz_fragment00",
         "localfile":"castles.tgz_fragment00",
         "md5sum":"ba5baa9af8d5ec9fdc58d46018b8767e"},
        {"boxfile":"s398vj4fka2wtmvrln2l219ueq9cl77c.tgz_fragment01",
         "localfile":"castles.tgz_fragment01",
         "md5sum":"89a6e394e2730145b0a62cbac47739e4"},
        {"boxfile":"dwnfdpqoyi2lc3sfupn6cw1ih6v8f4i1.tgz_fragment02",
         "localfile":"castles.tgz_fragment02",
         "md5sum":"cac2cbaea429855da312096d016bcfbd"},
        {"boxfile":"0ha44hus3o77vql0mlx21908qoe4w1gf.tgz_fragment03",
         "localfile":"castles.tgz_fragment03",
         "md5sum":"77999fcf6ae27b46ab56125fe67587f9"},
        {"boxfile":"j9sma0sctk74yrnuwk8k103uf2tw4h41.tgz_fragment04",
         "localfile":"castles.tgz_fragment04",
         "md5sum":"d592c9a4ab59165fe7dc599d78ecc9fa"}
    ],
    "boxurlprefix":"https://byu.box.com/shared/static/",
    "whole":{"localfile":"castles.tgz","md5sum":"94a470ba1f7b6cb0611bddb9c635e6ed"}
}       
 
def getmd5(fname):
    md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for buf in iter(lambda: f.read(8192), b''):
            md5.update(buf)
    return md5.hexdigest()
 
def download_url(url, output_path, desc):
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True,
                        miniters=1, desc=desc)
    with open(output_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
 
def download_and_verify(full_url,local,md5sum,desc):
    NUM_RETRIES = 3
    success = False
    for _ in range(NUM_RETRIES):
        #print(full_url,local,md5sum)
        download_url(full_url,local,desc)
        print('Verifying integrity... ',end='',flush=True)
        md5 = getmd5(local)
        success = md5 == md5sum
        if not success:
            print(f'MD5 checksums didn\'t match!\n  Expected {md5sum}, got {md5}')
        else:
            print('successful.')
            break
    return success
 
if __name__ == '__main__':
    meta = METADATA
 
    box_prefix = meta['boxurlprefix']
    for i,f in enumerate(meta['fragments']):
        lcl,nf = f['localfile'],len(meta['fragments'])
        status = download_and_verify(box_prefix+f['boxfile'],f['localfile'],
                                     f['md5sum'],f'{lcl} [{i+1}/{nf}]' )
        if not status:
            print(f'Download of fragment {i+1}/{nf} unsuccessful!!  Exiting...')
            sys.exit(-1)
    local_files = [f['localfile'] for f in meta['fragments']]
 
    print('Assembling fragments into single file... ',end='',flush=True)
    final = meta['whole']['localfile']
    outf = open(final,'wb')
    subprocess.run(['cat', *local_files], stdout=outf)
 
    md5 = getmd5(final)
    if md5 == meta['whole']['md5sum']:
        print('SUCCESS!!')
 
        print('Cleaning up... ',end='',flush=True)
        for f in local_files:
            os.remove(f)
        print('done.')
    else:
        print('MD5 doesn\'t match!!')

