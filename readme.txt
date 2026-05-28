ffmpeg-7.1-full_build.zip が必要です。
C:\ProgramDataへ配置した後に
C:\ProgramData\ffmpeg\bin をwindowsのシステム環境変数Pathへ登録
スタートメニュー→sysdm.cpl→詳細設定→環境変数→システム環境変数→Path
https://github.com/GyanD/codexffmpeg/releases/tag/7.1

metadata仕様（csv）
file_nameは2行目以降のデータ形式を参照するのでGUI上ではaudioとなる（仕様）
file_name,text
000.wav,どうしてこんなに荒々しい態度をとるんだ？
001.wav,こんにちは、今日はいい天気ですね。


Attentionバックエンド切り替え
--attention-backend sdpa      # デフォルト、PyTorch標準
--attention-backend flash2    # 要: pip install flash-attn（最速）
--attention-backend eager     # フォールバック用

勾配チェックポイント（VRAM節約）
--grad-checkpoint
VRAMを最大40%削減できます。速度は約20%低下しますがOOMが出る場合に有効です。

EMA（推論品質向上）
--ema-decay 0.9999

チェックポイント保存時に checkpoint_XXXXX_ema.pt も自動生成されます。
拡張オプティマイザ
--optimizer lion              # 要: pip install lion-pytorch
--optimizer ademamix          # 要: pip install optimi
--optimizer sgd
--optimizer adamw             # fused=True で高速化済み
--optimizer muon              # 元々の実装（変更なし）

勾配クリッピング調整
--clip-grad-norm 1.0    # デフォルト
--clip-grad-norm 0      # 無効化


Early Stopping（早期停止）実装
これを使うと学習失敗後の無駄なGPU消費を抑制できます。
valid lossを監視
    ↓
前回より悪化した？
    ├─ No  → patience(猶予)カウンタをリセットして続行
    └─ Yes → カウンタ+1
                ↓
           カウンタが閾値(例:1epoch分)を超えた？
                ├─ No  → 続行（経過観察中）
                └─ Yes → 学習停止 + その時点で保存

ema版モデルの保存を標準化＆full版をオプションへ移行
full版は学習履歴（追加の学習再開時に有効）を含めるのでデータ容量が大きい
条件保存されるファイル
EMA有効・デフォルトcheckpoint_XXXXXXX_ema.pt（推論用）
EMA有効・--save-fullcheckpoint_XXXXXXX_ema.pt + checkpoint_XXXXXXX_full.pt
EMA無効checkpoint_XXXXXXX.pt（フル版のみ・従来通り）


学習モデルの保存先をcheckpointsフォルダへ変更
モデルが無い場合は自動ダウンロードを行う
Irodori-TTS/
└── checkpoints/
    │
    ├── [TTSモデル] ※以下どちらの形式でも認識される
    │   │
    │   ├── 手動配置の場合（任意のフォルダ名）
    │   │   └── MyFolder/
    │   │       └── model.safetensors  ✅
    │   │
    │   └── HF自動DL形式（hf_hub_downloadが生成するフォルダ名）
    │       └── models--Aratako--Irodori-TTS-500M/
    │           └── snapshots/xxxx/
    │               └── model.safetensors  ✅
    │
    ├── codecs/               ← DACVAEコーデックキャッシュ（自動生成）
    │   └── dacvae-watermarked.pth
    │
    └── tokenizers/           ← トークナイザーキャッシュ（自動生成）
        └── models--llm-jp--llm-jp-3-150m/
            └── ...


 学習タブ（主要機能）
機能詳細
プリセット管理configs/ フォルダのYAMLをプルダウン選択、保存・更新ボタン付き
デフォルト設定train_v1.yaml の値を自動反映
コマンドプレビュー設定変更のたびにリアルタイムでCLIコマンドを表示
グラフ自動更新3秒ごとにログをポーリングし gr.LinePlot でLoss/LR曲線を描画
TensorBoard対応学習終了時に logs/tensorboard/ へイベントファイルを自動生成。CSVも同時保存
各項目の説明文全設定に日本語の簡易説明を記載
オプション機能Early StoppingはチェックボックスでON/OFF


🔄 チェックポイント変換タブ
プロジェクト配下の .pt をプルダウン選択して .safetensors に変換。
注意点： pandas が必要です（uv add pandas または pip install pandas）
gr.LinePlot はDataFrameを受け付けるため必須です。

Aratako_Irodori-TTS-500Mのmodelを使って追加学習する機能を追加
従来のスクラッチ学習もそのまま--resumeを使わない場合で残す

スライス＆キャプション機能を完全新規で実装
モデルが足りない場合はcheckpointsへ自動ダウンロード

モデルマージ機能実装（まだ動作未確認）


🎭 絵文字キャプションが追加され、以下の操作ができます。
入力CSVとwavフォルダを指定した後、APIプルダウンで選択します。
**「なし（従来通り）」を選ぶとLLM処理なしでtextをそのままcaptionカラムにコピーします。
「LM Studio（ローカル）」を選ぶとlocalhost:1234に自動接続、APIキー不要です。「Groq」「OpenAI」「Together AI」**を選ぶとAPIキー入力欄が現れ、デフォルトモデル名が自動入力されます。
実行すると各wavから音響特徴量を抽出してLLMに送信し、EMOJI_ANNOTATIONSルールに従った絵文字付きcaptionカラムが元CSVに追記されます。

必要な追加パッケージ
pip install librosa openai

librosaが音響解析、openaiがLM Studio含む全APIの統一クライアントとして機能します。
（カラムて何だよ？→csvで列のことです）