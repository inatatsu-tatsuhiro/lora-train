# lora-train-image

SaladCloud / Kelpie で SDXL LoRA 学習（ミラ）を回すためのコンテナイメージ置き場です。
イメージに入っているのは汎用コードだけです（PyTorch + kohya sd-scripts + Kelpie + 学習エントリポイント）。
モデル・データセット・認証情報は一切含まれていません。

## あなたの手順（1回だけ）

1. GitHub で **Public** リポジトリを新規作成する（名前の例: `lora-train-image`）。
2. このフォルダの中身を全部アップロードする。
   - リポジトリのページで「Add file」→「Upload files」を開き、解凍したフォルダの中身（`.github` フォルダを含む）をまとめてドラッグ＆ドロップ → コミット。
   - `.github/workflows/build.yml` が正しいパスに入っていることが大事です。
3. アップロードすると GitHub Actions が自動でビルドを開始します。「Actions」タブで緑色のチェックが付けば成功です（大きなイメージなので 15〜30 分くらいかかります）。
4. ビルド成功後、発行されたパッケージを公開に切り替える。
   - リポジトリ（またはプロフィール）の「Packages」から `lora-train` を開く →「Package settings」→「Change visibility」→ **Public**。
5. リポジトリの URL を Muse に伝える。

## 中身

- `Dockerfile` — PyTorch 2.7.1 / CUDA 12.6 ランタイム + sd-scripts（Mac の学習環境と同じコミットに固定）+ Kelpie 0.6.0。
- `train_entry.sh` — 学習の実行スクリプト。条件は Mac 実行時と同一（dim 32、12 エポック、3 エポックごとに保存など）。クラウド用に追加したのは中断時の自動レジューム（`--save_state` / `--resume`）だけです。ベースモデルは開始時に HuggingFace の公式リポジトリ（cagliostrolab/animagine-xl-4.0）から取得します。
- `.github/workflows/build.yml` — GHCR へ `ghcr.io/<あなたのアカウント>/lora-train:latest` をビルド＆プッシュします。
