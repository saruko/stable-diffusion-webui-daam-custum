# DAAM 拡張機能(Stable Diffusion WebUI Forge Neo 対応版)

プロンプト中の各単語が、生成画像のどの部分に効いているかを **ヒートマップ** で
可視化する拡張機能です。[DAAM](https://github.com/castorini/daam)
(Diffusion Attentive Attribution Maps)を WebUI 用に移植したものです。

オリジナルは AUTOMATIC1111 WebUI 向けで 2024 年初頭に更新が止まっていました。
本フォークは **[Stable Diffusion WebUI Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic)**
(ComfyUI 系に書き直された `backend`)で動くよう、内部を作り直した現代版です。

対応モデル: **SD 1.x** / **SDXL** / **Flux** / **Chroma** / **Lumina 2 (NextDiT)**

---

## インストール

このフォルダ一式を Forge Neo の `extensions` フォルダに置いて、UI を再起動してください。

```
<Forge Neo>/extensions/sd-webui-daam/
```

例(StabilityMatrix 経由でインストールしている場合):
`<StabilityMatrix のデータフォルダ>\Packages\<Forge Neo のパッケージ名>\extensions\sd-webui-daam\`

依存関係(matplotlib)は無ければ自動でインストールされます。

---

## 使い方

1. **設定 → 画像保存 →「生成した画像を常に保存する」を ON** にします。
   （画像が保存されるタイミングでヒートマップを生成する仕組みのため必須です）
2. txt2img / img2img 画面の **「Attention Heatmap」** アコーディオンを開きます。
3. **Attention texts** に、プロンプトに含まれる単語を入力します。
4. 通常どおり生成します。

生成すると、元画像に加えて、指定した単語ごとにヒートマップを重ねた画像が出力され、
通常の出力フォルダに `元ファイル名_単語.png` という名前で保存されます。

### Attention texts の書き方

- **カンマ区切り** で複数指定できます。例: `cat, sunglasses, beach`
- カンマで区切らず並べた語は **1 つの連続した語** として扱われます。
  - `cat` → プロンプト中の `cat` トークンすべてにマッチ
  - `cute cat` → `cute cat` という並びのトークンだけにマッチ
- プロンプトに含まれない単語を指定した場合、その単語はヒートマップなし
  (オーバーレイなしの元画像)として出力され、コンソールに警告が出ます。

### オプション

| オプション | 説明 |
|---|---|
| **Hide heatmap images** | ヒートマップ画像を UI ギャラリーに追加しない |
| **Do not save heatmap images** | ギャラリーには出すがファイル保存はしない |
| **Hide caption** | 画像に単語キャプションを描き込まない |
| **Use grid** | 複数のヒートマップを 1 枚のグリッドにまとめる(grids フォルダへ保存) |
| **Grid layout** | グリッドの並べ方(Auto / Prevent Empty Spot / Batch Length As Row) |
| **Heatmap blend alpha** | ヒートマップの重ね合わせ濃度(0〜1) |
| **Heatmap image scale** | 出力画像のスケール(0.1〜1.0) |
| **Trace each layers** | 注意ブロックごと(UNet: `IN##`/`MID`/`OUT##`、Flux: `D##`、Lumina: `L##`)に個別のヒートマップを出す |
| **Use layers as row** | レイヤー別ヒートマップをグリッドの行として並べる |

---

## 出力例

プロンプト: `A photo of a cute cat wearing sunglasses relaxing on a beach`

Attention texts: `cat, sunglasses, beach`

出力: 元画像 / cat / sunglasses / beach

<img src="images/00006-2623256163.png" width="150">
<img src="images/00006-2623256163_cat.png" width="150">
<img src="images/00006-2623256163_sunglasses.png" width="150">
<img src="images/00006-2623256163_beach.png" width="150">

---

## 仕組み(Forge Neo 移植の要点)

オリジナルは `ldm` の `CrossAttention.forward` を直接書き換えていましたが、Forge Neo
は `ldm`/`sgm` を使わず独自 `backend` を持ちます。本版では:

### SD 1.x / SDXL (UNet + CLIP)

- 注意の取得を Forge **公式の ModelPatcher API**(`unet.set_model_attn2_replace`)
  経由で行います(内部の monkey-patch をしません)。
- 出力は本物の `backend.attention.attention_function` で計算するため、
  **生成画像は DAAM 無効時と同一** です。ヒートマップ用の softmax だけを別途取得します。
- トークナイズは Forge の `ClassicTextProcessingEngine` を利用します。
- 条件側(cond)の注意だけを `cond_or_uncond` で選び、空間サイズは `original_shape`
  から復元するため、バッチサイズや hires-fix にも追従します。
- UNet パッチャは毎回クローンに対して適用し、Forge が生成ごとにリセットするため
  手動のアンフックは不要です。

### anima (Cosmos-Predict2 DiT)

- anima の `SelfCrossAttention` は attn2 パッチ辞書を参照しないため、
  `block.cross_attn` モジュールに直接 `register_forward_pre_hook` を挿入します。
- テキスト埋め込みは Qwen hidden states → T5 トークン ID で条件付けされるため、
  `AnimaPromptAnalyzer` が T5 トークナイザーを使って単語位置を特定します。
- フックは生成後の `postprocess` で明示的に `remove()` します。

### Flux / Chroma (Double-Stream DiT) ← **今回追加**

**アーキテクチャ上の違い**: Flux/Chroma は cross-attention を持たず、
テキストトークンと画像パッチトークンを **連結した joint self-attention**
(`DoubleStreamBlock`) を使います。

**方法論**: 各 `DoubleStreamBlock` に `forward_pre_hook` を挿入し、
本体 forward と **同一の手順**（modulation → qkv 投影 → QKNorm → RoPE 適用）
で query/key を再計算。そのうえで

```
attn_img_to_txt = softmax( q_img @ k_txt^T / sqrt(d) )
```

「画像トークン(query) → テキストトークン(key)」の部分行列だけを抽出します。
ヘッドごとにループして平均することで、`seq_img × seq_total` の巨大行列を
メモリに確保せずに済みます（1024px 生成で約 1.8GB の削減）。

- テキスト位置は `T5TextProcessingEngine` を使用（末尾パディングは除去）。
- Chroma は Flux と同じ `DoubleStreamBlock` を共有するため、**同一 Tracer で対応**。
- ブロックキーは `D00`, `D01`, ... (`D` = double block)。

### Lumina 2 / Neta Lumina (NextDiT) ← **今回追加**

**アーキテクチャ上の違い**: Lumina 2 はキャプション(cap)トークンと
画像パッチを連結した `[cap | img]` シーケンスを全層共通の
`JointTransformerBlock.attention` に流します。Flux の双方向とは異なり、
GQA (Grouped Query Attention) と独自 RoPE が適用されます。

**方法論**: 2 段階のフックを使います:
1. `context_refiner[0]` の pre-hook でキャプション長(`cap_len`)を毎ステップ記録
2. 各 `layers[i].attention` の pre-hook で q/k を再計算し、
   `q[cap_len:]`（画像 query）と `k[:cap_len]`（キャプション key）の
   部分積 softmax を抽出

Gemma2 のシステムテンプレート（Neta 指示文）は推論時に自動前置されるため、
`LuminaPromptAnalyzer` も同じ `process_template()` を呼んでからトークナイズし、
単語位置のズレを防ぎます。

- ブロックキーは `L00`, `L01`, ... (`L` = layer)。

### 共通の設計原則

| 項目 | 方針 |
|---|---|
| 生成画像への影響 | ゼロ。フックは読み取り専用で、失敗は握りつぶす |
| 空間解像度 | latent 1/8 × patch 2×2 = `ceil(H/16) × ceil(W/16)` で正確算出 |
| CFG 対応 | `transformer_options["cond_or_uncond"]` で条件側のみ集計 |
| 集計方法 | 各タイムステップ・各ブロックの attention を平均し、`64×64` latent サイズに bicubic upsample して合算 |

---

## 注意点・制限

- **SD3 は非対応** です。Forge Neo 自体が SD3 をサポートしておらず、
  対応するモデルエンジンが存在しないため追加不可能です。
- 対応外のモデルで生成した場合はコンソールに `unsupported model` と表示して
  自動的にスキップします(生成自体は通常どおり行われます)。
- ヒートマップ取得のぶん、DAAM 有効時は生成がやや遅くなります。
- SD 1.5 で最も安定します。アーキテクチャによって精度は変わります。
- Flux/Lumina のヒートマップは joint attention の部分行列抽出のため、
  SD 系の cross-attention 直取得より若干スムーズさが落ちることがあります。
- ヒートマップ生成には画像の自動保存が必要です(上記「使い方」1 を参照)。
- hires-fix で **元プロンプトと hires プロンプトのトークン数が異なる** 場合、
  hires パスのヒートマップは出ないことがあります(通常の同一プロンプト運用では問題ありません)。
- Lumina 2 のシステムテンプレート(Neta テンプレート)の内容が WebUI 設定で変更された場合、
  `Attention texts` の単語位置が変化する可能性があります。

---

## ライセンス

オリジナル DAAM / 本拡張のライセンスは [LICENSE](LICENSE) を参照してください。
