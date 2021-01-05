# Have Fun Storming the Castle(s)!

This is the official code repository for our WACV 2021 paper.

## Dataset

| Images | Castles | Countries | Construction Dates |
| :---: | :----: | :----: | :----: |
| 772,927 | 2412 | 93 | 85% coverage |

### Download

We've provided a [python script](download.py) to download the data. The download process requires ~100 GB of disk space. Run the download script:
```bash
python download.py
```
This will download several file fragments and recompile them into a single `castles.tgz` file, checking `md5` hashes along the way. Once the process is complete, you can remove the fragments:
```bash
rm *.tgz_fragment*
```
Then extract the data from `castles.tgz`:
```bash
tar xzf castles.tgz
```

Additional metadata, such as location and construction dates, can be found in [metadata.json](metadata.json).

If you prefer to do the download manually, check `download.py` for remote urls and expected `md5` hashes.


## Experiments

All experiments use [PyTorch](https://pytorch.org/) version `1.6.0` and [PyTorch Lightning](https://www.pytorchlightning.ai/) version `0.9.0`. Image retrieval experiments also depend on [PyTorch Metric Learning](https://kevinmusgrave.github.io/pytorch-metric-learning/) version `0.9.93`.
