# gui_setting/

Gradio GUI（`gradio_app.py`）と `dataset_tools.py` が使う設定置き場です。

## 同梱ファイル（git管理）

| ファイル | 内容 |
|---|---|
| `system_prompt_emoji.txt` | 絵文字キャプション生成のシステムプロンプト初期テキスト（EMOJI_ANNOTATIONS仕様） |
| `system_prompt_voice_analysis.txt` | 音声分析キャプション生成のシステムプロンプト初期テキスト |
| `README.md` | このファイル |

- 中身を編集すると、その内容がそのままLLMのシステムプロンプトとして使われます。
- 別ファイルを使いたい場合は、GUIの「⚙️ 設定」タブでパスを指定してください（空欄ならこの初期ファイルが使われます）。
- CLIの場合は `dataset_tools.py emoji_caption --emoji-prompt-file <path> --voice-prompt-file <path>` で指定します。

## 生成されるファイル（git管理外・コミット対象外）

| ファイル | 内容 |
|---|---|
| `gui_settings_temp.json` | GUIのユーザー入力値・テーマ選択・システムプロンプトファイル指定の保存（次回起動時の復元用） |

- 削除すればすべての設定は初期状態に戻ります。
- `.gitignore` で `gui_setting/*_temp.json` は除外済みです。
