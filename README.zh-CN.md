# meetnote

[English](README.md) | **简体中文**

会议录音 → **转写稿** + **全文梗概** + **两页精华**，一条命令跑完。

- 转写：本地 faster-whisper（large-v3），有 NVIDIA 显卡自动走 GPU。1 小时录音大约几分钟，无需 ffmpeg。
- 总结：调用大模型生成，默认用 DeepSeek，也支持其他 OpenAI 兼容服务和 Claude。

## 安装（一次性）

需要 Windows、Python 3.10+；有 NVIDIA 显卡会快很多。

```powershell
git clone https://github.com/YuLiuPotato/meetnote.git
cd meetnote
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

安装脚本会在 `meetnote\.venv` 建独立环境，并在 `%USERPROFILE%\.local\bin` 放一个 `meetnote.cmd`（请确认这个目录在 PATH 上）。Whisper 模型（约 3GB）第一次运行时自动下载。

### 配置文件 `.env`

API key、端口、访问码都写在 `meetnote\.env` 里（安装时从 `.env.example` 复制生成），不需要设置系统环境变量。用记事本打开，填上：

```ini
DEEPSEEK_API_KEY=sk-你的key
```

界面语言（提示信息、`--help`、评估报告和输出文件名）默认跟随系统显示语言；在 `.env` 里设 `MEETNOTE_LANG=zh` 或 `MEETNOTE_LANG=en` 可以指定。它和纪要用什么语言写无关。

`.env` 的查找顺序：环境变量 `MEETNOTE_ENV_FILE` 指定的文件 → 当前目录的 `.env` → `meetnote\.env`。已经存在的同名环境变量优先于文件。`.env` 已在 `.gitignore` 里，不要发给别人。

**总结需要配置一个大模型（`--backend auto` 会按下表顺序选第一个配置好的）：**

| 后端 | `.env` 里填 | 默认模型 |
|---|---|---|
| `deepseek`（推荐） | `DEEPSEEK_API_KEY`（在 platform.deepseek.com 申请） | `deepseek-chat`，可用 `--model deepseek-reasoner` |
| `openai` | `MEETNOTE_BASE_URL` + `MEETNOTE_API_KEY` + `MEETNOTE_MODEL`，适用于 Kimi、通义、智谱等 OpenAI 兼容服务 | 自己指定 |
| `claude-cli` | 不用填；安装独立版 Claude Code，`claude` 在 PATH 上并已登录 | 你的 Claude Code 设置 |
| `claude-api` | `ANTHROPIC_API_KEY` | `claude-opus-5-5` |

## 用法

```powershell
# 完整流程
meetnote run "D:\录音\周会.m4a"

# 加上术语和背景，识别和总结都更准
meetnote run 周会.m4a --terms "Kubernetes,OKR,张伟" --context "产品团队的季度路线讨论"

# 知道参会人数就写上，区分说话人会准很多
meetnote run 周会.m4a --speakers 3

# 只转写 / 对已有转写稿重新生成总结（比如改了提示词之后）
meetnote transcribe 周会.m4a
meetnote summarize 周会_纪要

# 把说话人标签换成姓名，再重新生成纪要
meetnote rename 周会_纪要 S1=张三 S2=李四
meetnote summarize 周会_纪要
```

输出默认放在当前目录的 `<文件名>_纪要\` 下（界面语言为中文时；英文时是 `<文件名>_notes\`，文件名为 `transcript.txt`、`highlights.md` 等）。已有文件的目录会沿用原来的命名。

```
周会_纪要\
  转写稿.txt     带时间戳和说话人的全文，如 “[01:23] S2: ……”；说话人一换就换行
  转写稿.srt     字幕格式，可以拖进播放器对照原音
  要点笔记.md    逐段整理的观点/事实/决定，每条带时间戳和评价对象
  全文梗概.md    按时间顺序还原整场会议，带出处时间戳
  两页精华.md    结论先行，控制在两页 A4，带出处时间戳
  核查记录.md    核查员发现并已修订的问题
  meta.json      时长、语言、来源文件等
```

### 多语言

- 录音语言默认**自动识别**，并且**逐段识别**：中英混杂的会议，中文部分出中文、英文部分出英文，不会互相翻译。识别不准时可用 `--language zh` 等固定。
- 纪要语言默认跟录音语言一致，可用 `--summary-lang zh` / `en` 指定（比如英文会议出中文纪要）。

### 区分说话人

转写时会自动区分说话人，每行标上 `S1`、`S2`……（按第一次开口的顺序编号）。纪要会按标签写“S1 认为……”，不会猜标签对应的真名；转写稿里有自我介绍或点名称呼时，会写成“S2（自称李明）”。

- **知道人数就填** `--speakers N`（网页上是“参会人数”）。不填时自动估计。
- **换成真名**：先看转写稿里每个标签说了什么，再用 `meetnote rename`（网页上是“说话人 · 填写姓名”）。同一个人被分成两个标签时，填同一个名字就会合并。
- 原理：sherpa-onnx 先找出语音段和换人的位置（pyannote segmentation-3.0），然后每 1.5 秒提取一次声纹（3D-Speaker CAM++，中英文训练），再按声音相似度聚类；发言很少的人（比如问答环节的提问者）只有声音相似时才会并到别人身上，否则单独成为一个说话人。Whisper 漏掉、但说话人模型检测到的语音会单独补识别。全部在 CPU 上运行，和显卡上的转写同时进行，不需要 Hugging Face token。模型约 35 MB，第一次使用时自动下载到 `%USERPROFILE%\.cache\meetnote\models`。
- 不想区分说话人：`--no-diarize`。
- 局限：两人同时说话时只标一个人；“嗯”“对”这类一两个字的插话容易标错；电话会议、远场麦克风效果会差一些。

### 总结流程与准确性

默认的 `grounded` 流程分四步，专门针对“张冠李戴”：

1. **逐段抽取**：把转写稿切成约 10 分钟一段，逐段整理要点，每条观点必须写明**评价对象**（哪个产品/版本/方案）和时间戳，对象不明就标“对象不明”。
2. **带出处撰写**：以要点笔记为依据写梗概和精华，每条结论后标 `[mm:ss]`。
3. **逐条核查**：另一次调用对照转写稿检查无依据、对象错误、归属错误、方向/数字错误、过度概括。
4. **按核查意见修订**。

`--pipeline simple` 是旧的单次生成，更快更省，但长会议容易混淆。

评估纪要忠实度（拆成单条陈述逐条对照转写稿打分，评审默认用 `deepseek-reasoner`）：

```powershell
meetnote check 周会_纪要
```

结果写在 `忠实度评估.md`。注意：LLM 评审本身也会出错（TofuEval 发现即使 GPT-4 做二元事实判断也不可靠），适合比较两个版本的相对好坏，不能当作绝对分数。

已经有转写稿时，`run` 会跳过转写直接总结；需要重新转写时加 `--force`。

## 常用参数

| 参数 | 作用 |
|---|---|
| `-o DIR` | 指定输出目录 |
| `--terms` | 人名、术语提示，逗号分隔，提升识别准确率 |
| `--context` | 会议背景，帮助 Claude 纠正术语、理解上下文 |
| `--only synopsis/highlights` | 只生成其中一份 |
| `--backend` | 强制选择总结后端：`deepseek` / `openai` / `claude-cli` / `claude-api` |
| `--model` | 指定模型名 |
| `--whisper-model medium` | 更快、稍不准的转写模型 |
| `--device cpu` | 没有显卡时使用（会慢很多） |
| `--language zh` | 固定录音语言（默认自动识别） |
| `--speakers 3` | 参会人数（默认自动估计） |
| `--no-diarize` | 不区分说话人 |
| `--summary-lang en` | 纪要语言（默认跟录音一致） |
| `--pipeline simple` | 单次生成，快但不如默认流程准确 |

`.env` 里的 `MEETNOTE_CLAUDE` 可以指定 claude 可执行文件的路径。

## 网页服务（给同事用）

在一台有 NVIDIA 显卡的机器上运行网页服务，同事用浏览器上传录音、查看和下载纪要，不需要安装任何东西。页面右上角可以切换中文 / English，默认跟随浏览器语言。

- 任务按顺序排队，同一时间只转写一个，1 小时录音大约 3–5 分钟。
- 每个任务有一个不可猜测的链接（`/#job=…`），可以“复制分享链接”发给其他同事。
- 原始录音转写完就删除，只保留文字结果；结果默认保留 30 天（`--keep-days`）。
- 总结失败时转写稿会保留，页面上可以一键“重新生成总结”。
- 服务重启后，没跑完的任务会自动重新排队。

### 1. 配置（一次性）

在 `meetnote\.env` 里填写（key 只放在服务器上，同事不需要）：

```ini
DEEPSEEK_API_KEY=sk-你的key
# 可选：访问码，打开网页时需要输入
MEETNOTE_ACCESS_CODE=自己定一个
# 可选：换端口，默认 18765
# MEETNOTE_PORT=18765
```

改完 `.env` 需要重启服务。

### 2. 启动

双击 `start-server.cmd`，或在终端运行它。服务监听 `http://localhost:18765`（端口以 `.env` 为准），数据存在 `%USERPROFILE%\meetnote-data`。启动日志第一行会显示读取的是哪个 `.env`。

开机自动启动（登录后会弹出一个服务窗口，关掉窗口就是停止服务）：

```powershell
schtasks /Create /TN meetnote /SC ONLOGON /TR "<meetnote 所在目录>\start-server.cmd"
```

如果用笔记本做服务器，记得在电源设置里把“接通电源时睡眠”改成“从不”。

### 3. 让同事访问：Tailscale（推荐）

先在这台机器上安装并登录 Tailscale。用 Tailscale 自带的 HTTPS 反向代理把服务暴露到 tailnet 内部，不对公网开放：

```powershell
tailscale serve --bg 18765
```

（如果在 `.env` 里改了端口，这里也要用同一个端口。）

之后在 tailnet 内访问 `https://<机器名>.<你的 tailnet>.ts.net`（`tailscale serve status` 会显示完整地址）。同事加入的方式二选一：

- **邀请进 tailnet**：Tailscale 管理后台 → Users → Invite users。
- **只共享这台机器**：管理后台 → Machines → 这台机器 → Share。对方用自己的 Tailscale 账号接受后，只能访问这一台机器。

同事需要在电脑或手机上装 Tailscale 并登录。停止共享：`tailscale serve --https=443 off`。

### 备选：办公室局域网

如果大家在同一个局域网，可以不用 Tailscale：在 `.env` 里设 `MEETNOTE_HOST=0.0.0.0` 后启动，同事访问 `http://<这台电脑的局域网 IP>:18765`。另外需要管理员在 Windows 防火墙放行这个端口（入站，专用网络）。局域网方式是 http 明文传输，建议同时设置访问码。

## 自定义输出格式

两份总结的结构和要求写在提示词文件里，直接改即可（安装是 editable 模式，改完立刻生效）：

- `meetnote\prompts\synopsis.md`：全文梗概
- `meetnote\prompts\highlights.md`：两页精华

## 已知限制

- 说话人是自动区分的，会有少量错标（见“区分说话人”）；纪要里重要的“谁说的”请对照转写稿核实。
- 8GB 显存只够同时跑一个转写进程：网页服务运行时再用命令行转写，会因显存不足变得非常慢。
- 转写稿会发送给你配置的大模型服务商，涉及敏感会议内容时请注意。
- 人名、机构名、中英混杂的术语容易识别错。用 `--terms` 能改善，两页精华的文末也会列出需要核对的名称。
- 微信等应用的临时文件会被自动清理，建议先把录音另存出来再处理。

## 许可与致谢

代码以 [MIT](LICENSE) 许可发布。用到的模型和库：[faster-whisper](https://github.com/SYSTRAN/faster-whisper)（Whisper large-v3）、[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)、[pyannote segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)、[3D-Speaker CAM++](https://github.com/modelscope/3D-Speaker)。模型在第一次使用时下载，不随本仓库分发，各自遵循其许可。准确性改进与评测过程见 [docs/准确性改进与评测_2026-11-04.md](docs/准确性改进与评测_2026-11-04.md)。
