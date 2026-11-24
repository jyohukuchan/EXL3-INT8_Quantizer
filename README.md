# EXL3-INT8 Quantizer

EXL3-INT8 mul24 形式のモデルを生成する、独立した量子化ツールです。ROCm、CUDA のネイティブ拡張と Triton バックエンドを選択できます。`shelf` のビルドや実行ファイルは不要です。

出力は mul1 codebook の multiplier `0x00DCD12D` を使用します。一般的な EXL3 の別の multiplier や codebook とは互換とみなせません。変換時には拡張の multiplier を検査し、出力を `validate` で検証します。

## 準備

Python 3.10.11 以降と GPU に対応した PyTorch を用意し、依存パッケージをインストールしてください。ROCm のネイティブ拡張には ROCm と HIP のビルド環境、CUDA のネイティブ拡張には CUDA ツールキットが必要です。Triton を使う場合は対応する Triton も必要です。

```bash
python -m pip install numpy rich typing_extensions safetensors ninja pillow pyyaml marisa-trie pydantic llguidance tokenizers huggingface_hub transformers triton
python quantize.py build-ext --backend triton
```

ネイティブ拡張には `--backend rocm` または `--backend cuda` を指定します。`auto` は PyTorch の ROCm または CUDA を検出します。ROCm ビルドの既定の対象は `gfx1201` です。

## 変換と検証

```bash
mkdir -p work
python quantize.py convert --backend triton --model /path/to/model --output /path/to/output --work work --bpw 4.0
python quantize.py validate /path/to/output
```

`--threads`、`--seed`、`--calibration`、`--cal-rows`、`--cal-cols` で変換と calibration を調整できます。`--shard-size` は出力 shard の最大サイズを MB で指定します。出力には実行条件を記録した `quantize_report.json` が含まれます。

`--low-disk` は中間データの保存量を抑えます。`--hf-repo OWNER/MODEL` と組み合わせると Hugging Face の shard を必要に応じて取得します。`--model` はモデル用のローカルディレクトリ、`--hf-revision` は取得する revision です。`--checkpoint-dtype {fp32,bf16,fp16}` と `--no-checkpoint` でも容量を調整できます。checkpoint を無効にすると再開用の checkpoint は保存されません。

`--adapter` に PEFT LoRA のローカルディレクトリまたは Hugging Face リポジトリを指定すると、重みの読み込み時にマージします。リモートの revision は `--adapter-revision` で指定します。対応しない adapter 設定はエラーになります。

## Qwen3.5 系の KL 蒸留

Qwen3.5 系の dense モデルでは、量子化後に `suh`、`svh` とテキスト側の Norm を調整できます。量子化に使った BF16 モデルを教師に指定します。入力のトレリスとビット配分を保持し、結果を新しいディレクトリへ書き出します。

```bash
python quantize.py distill --model /path/to/bf16 --bundle /path/to/quantized --output /path/to/distilled --seed 52
```

校正は同梱のテキストから 16 行 × 128 トークンを取り、温度 1 の KL ダイバージェンスで 32 ステップ学習します。既定では BF16 モデル全体を GPU に配置します。`--gpu-linear-gib 6` を加えると主幹の線形重みを最大 6 GiB まで GPU に常駐させ、残りをホスト RAM から投影ごとに転送します。埋め込み、出力層、活性と一時領域の VRAM は別途必要です。`distill_report.json` に条件と損失を記録します。

## Google Colab

GPU ランタイムを選択し、以下をセルで実行します。モデル、作業領域、出力先には十分な空き容量を確保してください。非公開モデルでは Hugging Face の認証も必要です。

```bash
!git clone https://github.com/jyohukuchan/EXL3-INT8_Quantizer.git
%cd EXL3-INT8_Quantizer
!pip install numpy rich typing_extensions safetensors ninja pillow pyyaml marisa-trie pydantic llguidance tokenizers huggingface_hub transformers triton
!python quantize.py build-ext --backend triton
!mkdir -p model work
!python quantize.py convert --backend triton --model model --output output --work work --bpw 4.0 --low-disk --hf-repo OWNER/MODEL
!python quantize.py validate output
```

## ライセンスと由来

取り込み元とライセンスは [PROVENANCE.md](PROVENANCE.md) および [MIT ライセンス](exl3_int8/EXLLAMAV3-LICENSE.txt) を参照してください。
