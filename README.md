# #11 atmaCup
- 2021-07-09 ~ 2020-07-21 に行われた [#11 [初心者歓迎! / 画像編] atmaCup](https://www.guruguru.science/competitions/17) のリポジトリです。結果は [Public 4th / Private 5th](https://www.guruguru.science/competitions/17/leaderboard) でした。
- フレームワークは PyTorch で、実装は [pytorch-image-models(timm)](https://github.com/rwightman/pytorch-image-models) と [pytorch-pfn-extras](https://github.com/pfnet/pytorch-pfn-extras) に完全に依存しています。おそらく pfn-extras 使いでなければ読めません。  

## 目次

* [解法概要](#解法概要)
* [ディレクトリ構成](#ディレクトリ構成)
* [実行手順](#実行手順)
    * [環境](#環境)
    * [準備](#準備)
    * [学習](#学習)
    * [推論](#推論)
* [その他補足](#その他補足)
    * [ソースコードの構成について](#ソースコードの構成について)
    * [結果の再現性について](#結果の再現性について)
    * [出力等について](#出力等について)
    * [pytorch-pfn-extras 使いでない方へ](#pytorch-pfn-extras使いでない方へ)
## 解法概要

詳細は discussion で公開しています [[link](https://www.guruguru.science/competitions/17/discussions/a768baf7-a805-46bb-ae29-88c92fd30fe6/)]

3行まとめ：

* SimSiam による事前学習
* Classication / Regression それぞれのタスクで Fine-tuning
* 後処理を行ったうえで Weight Optimization

## ディレクトリ構成

```
.
├── input
|     └── atmaCup-11       # コンペデータを置く場所
├── output                 # 学習結果の出力先
└── src                    # preprocess, training, inference 等の code
```

`./src` 下の構成については[その他補足](#その他補足)に記載。

## 実行手順

**以下ではスクリプトの実行を `./src` ディレクトリで行ってください。**

### 環境
#### GPU
* TitanRTX(主にSimSiam と重い model の学習に使用)
* GTX1080Ti(主に軽い model の学習と推論に使用)

batch size を落とす・Gradient Accumulation を使用する 等を行えば VRAM 容量が少なくても動かせると思います。

#### Python & cuda
- Python 3.8.6
- CUDA 10.2 (CUDA driver 440.33.01)

#### 主要なライブラリ
* 抜け漏れがあるかもしれないです
* 古すぎるとかでなければ Version が一致しなくても動くと思います

| Name           | Version |  
|:--------------:|:-------:|
| albumentations | 1.0.0   |
| joblib         | 1.0.1   |
| lightly        | 1.1.16  |
| matplotlib     | 3.4.2   |
| numpy          | 1.20.3  |
| opencv-python  | 4.5.2.54|
| optuna         | 2.8.0   |
| pandas         | 1.2.4   |
| pytorch-pfn-extras | 0.4.1 |
| PyYAML         | 5.4.1   |
| scikit-learn   | 0.24.2  |
| scipy          | 1.6.3   |
| timm           | 0.4.12  |
| torch          | 1.9.0   |
| torchvision    | 0.10.0  |
| tqdm           | 4.61.0  |


### 準備
#### コンペティションデータの格納
ダウンロードして `./input/atmaCup-11` に解凍、photos.zip もその場で解凍してください。  
以下のような構成になることを想定しています。

```
.
├── input
|     └── atmaCup-11
|             ├── photos
|             ├── atmaCup#11_sample_submission.csv
|             ├── materials.csv
|             ├── techniques.csv
|             ├── test.csv
|             └── train.csv
.
.
```

#### 前処理

以下を実行。
~~~
$ python preprocess.py
~~~

各画像のサイズ等が格納された `img_info.csv`, データセット全体の(概算の)channel ごとの統計値が計算された `stats_by_data.csv`、`train.csv` に Cross Validation のための分割(`fold` 列)が追加された `train_sgkf-5fold.csv` が `./input/atmaCup-11` 下に生成されます。 

### 学習
#### 事前学習

まず `ResNet18-D`, `ResNet34-D`, `ResNet50-D`, `Fast-ResNeSt50-D_1s4x24d` の 4モデルについて SimSiam による事前学習を行います。
GPU に乗らない場合は gradient accumulation 使用を検討してください。

~~~
$ python train_simsiam.py -cfg exp_config/000.yml  # resnet18d
$ python train_simsiam.py -cfg exp_config/001.yml  # resnet34d
$ python train_simsiam.py -cfg exp_config/002.yml  # resnet50d
$ python train_simsiam.py -cfg exp_config/003.yml  # resnest50d_1s4x24d
~~~

#### Fine-tuning

自動で 5fold の training を実行。Regression, Classification の各タスクで行うので 8種のモデルが出来ます。
前述の SimSiam の学習結果が以下のように `./output`下に出力されており、これらを読む込んで使います。  
(`ResNet18-D`, `ResNet34-D` は 150 epoch, `ResNet50-D`, `Fast-ResNeSt50-D_1s4x24d` は 200 epoch 時点の事前学習モデルを使用。)
```
.
├── output
|     ├── 000_resnet18d_simsiam
|     ├── 001_resnet34d_simsiam
|     ├── 002_resnet50d_simsiam
|     └── 003_resnest50d_1s4x24d_simsiam
.
.
```

##### Classification
~~~
$ python train.py -cfg exp_config/100.yml  # resnet18d
$ python train.py -cfg exp_config/101.yml  # resnet34d
$ python train.py -cfg exp_config/102.yml  # resnet50d
$ python train.py -cfg exp_config/103.yml  # resnest50d_1s4x24d
~~~

##### Regression
~~~
$ python train.py -cfg exp_config/200.yml  # resnet18d
$ python train.py -cfg exp_config/201.yml  # resnet34d
$ python train.py -cfg exp_config/202.yml  # resnet50d
$ python train.py -cfg exp_config/203.yml  # resnest50d_1s4x24d
~~~

### 推論

学習が完了していると `./output` 下に各学習結果のディレクトリが生成されているはずです。これらを読み込んで使用します。

```
.
├── output
|     ├── 100_resnet18d_cls
|     ├── 101_resnet34d_cls
|     ├── 102_resnet50d_cls
|     ├── 103_resnest50d_1s4x24d_cls
|     ├── 200_resnet18d_reg
|     ├── 201_resnet34d_reg
|     ├── 202_resnet50d_reg
|     └── 203_resnest50d_1s4x24d_reg
.
.
```

#### モデルごと
各学習結果の ディレクトリを指定する形で実行します。  

**!!注意!!：同じディレクトリ内に metric(今回は RMSE) での各 fold での best model が copy され、学習過程のチェックポイントは全て削除されます。**

同じディレクトリ内に各 fold での best model での予測結果、5-fold averaging 、oof prediction ( + classification の場合は logit の状態のもの)、各 fold での CV の結果の csv が出力されます。後処理を実施した上での予測結果です。

##### Classification
~~~
$ python infer.py -e ../output/100_resnet18d_cls
$ python infer.py -e ../output/101_resnet34d_cls
$ python infer.py -e ../output/102_resnet50d_cls
$ python infer.py -e ../output/103_resnet50d_1s4x24d_cls
~~~

##### Regression
~~~
$ python infer.py -e ../output/200_resnet18d_reg
$ python infer.py -e ../output/201_resnet34d_reg
$ python infer.py -e ../output/202_resnet50d_reg
$ python infer.py -e ../output/203_resnet50d_1s4x24d_reg
~~~

#### アンサンブル

以下を実行してください。

~~~
$ python ensemble.py -cfg exp_config/900.yml
~~~

Classification/Regression モデルのみでの averaging, 全モデル(8 model)での averaging、oputuna で weight optimization を行った結果、が出力されます。


## その他補足
### `./src` の構成について

少し補足しておくと、`./src` 下のディレクトリ・ファイルの中身はざっとこんな感じです。

```
.
├── src
|     ├── base_data         # コンペ問わず使いまわす dataset 等
|     ├── base_model        # コンペ問わず使いまわす model 等
|     ├── base_optimizer    # コンペ問わず使いまわす optimizer 等
|     ├── base_pfn_extras   # コンペ問わず使いまわす pfn-extras 関連
|     ├── utils             # その他の使いまわすコード
|     ├── data.py           # コンペ特有の dataset 等を作ったら書く
|     ├── model.py          # コンペ特有の model 等を作ったら書く
|     ├── global_config.py  # (コンペ特有の)全体的な設定などを記述
|     ├── preprocess.py     # コンペ特有の前処理
|     ├── train_simsiam.py  # SimSiam の学習
|     ├── train.py          # Fine-tuning の学習
|     ├── infer.py          # 推論
|     └── ensemble.py       # アンサンブル
.
.
```

`base_XXX` と `utils` は固定で、コンペで都度都度必要になったものは `model.py`, `data.py` 等に新しく追加します。コンペ終了後「また使いそうだな」というものは `base_XXX` に統合する運用です(例えば今回なら SimSiam のために書いた Dataset を終了後に統合しました)。  一応再現性を保つという名目で `model.py`, `data.py`, `global_config.py`, `train[_simsiam].py` は学習ごとに結果の出力先へコピーを取るようにしています。

`train.py` は基本使いまわしでコンペごとに一部(主にデータの読み込みの部分)を書き換えて使いますが、`infer.py`(, `ensemble.py`)は、指標等のせいで書き換える部分が多くなる場合がほとんどです(今回なら後処理の部分など)。

またこれは `pytorch-pfn-extras` のしかも `Config System` を使っている人にしか伝わらない話ですが、`config_types` の辞書は一旦各 `base_XXX` の `__init__.py` に作って置き、それらを `global_config.py` 内で読み込んで一つの辞書(`CONFIG_TYPES`)に統合しています。`data.py` や `model.py` で新しく作ったものについても `global_config.py` 内で追加します。

### 結果の再現性について

乱数等は固定するとともに `torch.backends.cudnn.deterministic` を True にしていますが、基本的に速度を優先して `torch.backends.cudnn.benchmark` を True にしているので実行ごとに結果が変わります(詳細：[Reproducibility — PyTorch 1.9.0 documentation](https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking))。

完全に再現性を取りたい場合は `torch.backends.cudnn.benchmark` を False にすれば(多分)行けるはずです。

### 出力等について

- このリポジトリは terminal での実行を前提としていますが、notebook に移植する場合は pfn-extras が出してくれるプログレスバーの表示がうまくいきません。もし移植するのであれば各 config yaml ファイルにある `ProgressBar` をコメントアウトし、`train.py` の 139行目にある `Evaluator` の引数 `progress_bar` を False にしてください。

- 学習の出力結果を一切上げていないので何が出てくるか補足しておくと、学習ログの json ファイル、指定したタイミングでの model の snapshot、loss・metric・lr を可視化した png ファイルです。ここらへんの設定は config yaml ファイル の `extensions` で指定しています。

### pytorch-pfn-extras使いでない方へ
特に `Config System` を使用しているせいで面食らう部分もあるかと思いますが、`train[_simsiam].py` を読んでいただけると流れ自体は basic な training loop とほぼ同じだとわかると思います(mixup とか gradient accumulation を入れたことでちょっとごちゃついてますが)。
manager と extensions の枠組みを使うことで素の training loop にあまり影響せずに前述の出力が出来るのが pytorch-pfn-extras の一番好きな所なので、興味がある方は是非使ってみてください！