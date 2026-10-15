@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo  Irodori-TTS 環境セットアップ
echo  Python 3.13→3.12→3.11 自動選択 / torch 2.9.1 + cu130 / uv 使用
echo ============================================================
echo.

rem ---------- 0. uv の確認 ----------
where uv >nul 2>nul
if not errorlevel 1 goto :uv_ok

echo [INFO] uv が見つかりません。winget でインストールを試みます...
winget install -e --id astral-sh.uv
set "PATH=%PATH%;%LOCALAPPDATA%\Microsoft\WinGet\Links;%USERPROFILE%\.local\bin"
where uv >nul 2>nul
if not errorlevel 1 goto :uv_ok

echo [ERROR] uv を自動インストールできませんでした。手動で導入後に再実行してください:
echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 ^| iex"
pause
exit /b 1

:uv_ok

rem ---------- 1. Python インタープリタの確認 ----------
rem .python-version のリスト（3.13 → 3.12 → 3.11）から uv が利用可能なものを自動選択する。
rem 一つも無い場合は uv sync 時に uv が 3.13 を自動ダウンロードするため、ここでは失敗扱いにしない。
echo [1/4] Python インタープリタを確認します（3.13→3.12→3.11 の順で自動選択）...
uv python find || echo [INFO] 一致するインタープリタが見つかりません。次の同期処理で uv が自動的に 3.13 をダウンロードします。

rem ---------- 2. 仮想環境の構築（量子化 extra を含む） ----------
echo [2/4] 仮想環境 .venv を構築し、依存をインストールします...
uv sync --extra quantization
if errorlevel 1 goto :err_sync

rem ---------- 4. 動作確認 ----------
echo [3/4] torch / CUDA の状態を確認します...
uv run python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); caps=[torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]; [print('GPU', i, torch.cuda.get_device_name(i), f'sm_{caps[i][0]}{caps[i][1]}') for i in range(torch.cuda.device_count())]; print('WARN: sm_75 未満の GPU があります。cu130 は未対応のため pyproject の index を cu126 に張替えて再実行してください') if any(c < (7, 5) for c in caps) else None"
if errorlevel 1 goto :err

echo [4/4] torchao / FFmpeg の状態を確認します...
uv run python -c "import torchao; print('torchao:', torchao.__version__)"
where ffmpeg >nul 2>nul
if errorlevel 1 echo [WARN] PATH 上に ffmpeg が見つかりません。音声読み込みでエラーが出る場合は winget install -e --id Gyan.FFmpeg 等で導入してください。

echo.
echo セットアップが完了しました。start.bat で起動できます。
pause
exit /b 0

:err_sync
echo [ERROR] uv sync に失敗しました。ネットワーク接続を確認してください。
echo         旧GPU（GTX 10系等・sm_75未満）の場合は pyproject.toml の index を cu126 に張替えて再実行してください。
pause
exit /b 1

:err
echo [ERROR] 処理中にエラーが発生しました。上の出力を確認してください。
pause
exit /b 1
