# 给朋友的零基础体验指南

这份指南用于在朋友自己的电脑上运行 NinjaMind 的 SFT、DPO、PPO、GRPO 四个模型并进行本地对话。整个过程不训练模型，也不需要租 GPU。

## 电脑要求

- Windows 10/11 64 位，或较新的 macOS；
- 至少 8 GB 内存，建议 16 GB；
- 至少 5 GB 可用磁盘空间；
- 能访问 GitHub 和 Python 软件包下载站；
- CPU 就能运行，不需要 NVIDIA 显卡或 CUDA。CPU 生成速度可能较慢，这是正常现象。

只需手动安装两个工具：

1. **Git**：下载代码；
2. **uv**：自动安装 Python 3.12、PyTorch、Gradio 和其他依赖。

不需要单独安装 Python、Anaconda、CUDA、Docker、Node.js 或 VS Code。

## Windows 安装与运行

### 1. 安装 Git 和 uv

打开开始菜单，搜索并打开 **PowerShell**，逐行运行：

```powershell
winget install --id Git.Git -e --source winget
winget install --id astral-sh.uv -e
```

如果电脑没有 `winget`，请从 [Git 官网](https://git-scm.com/downloads/win)安装 Git，
再在 PowerShell 运行 uv 官方安装命令：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安装完成后关闭 PowerShell，再重新打开。检查安装：

```powershell
git --version
uv --version
```

两条命令都显示版本号才继续。

### 2. 克隆仓库并安装运行环境

```powershell
cd $HOME\Desktop
git clone https://github.com/ziyang02/minimind_from_scratch.git
cd minimind_from_scratch
uv python install 3.12
uv sync --frozen --extra cpu --extra web
```

第一次安装需要下载 Python、PyTorch 和 Gradio，可能要几分钟。以后不需要重复安装。

### 3. 下载模型

打开项目的 [GitHub Releases 页面](https://github.com/ziyang02/minimind_from_scratch/releases/latest)，下载：

- `minimind_showcase_models.tar`
- `minimind_showcase_models.tar.sha256`（可选，用于校验文件）

默认会保存到“下载”文件夹。可用下面的命令检查下载文件是否完整；输出的哈希值应与 `.sha256` 文件中的长字符串相同：

```powershell
Get-FileHash "$HOME\Downloads\minimind_showcase_models.tar" -Algorithm SHA256
Get-Content "$HOME\Downloads\minimind_showcase_models.tar.sha256"
```

回到仓库目录并解压：

```powershell
cd $HOME\Desktop\minimind_from_scratch
tar -xf "$HOME\Downloads\minimind_showcase_models.tar" -C .
Get-ChildItem .\out\showcase
```

最后一条命令应显示四个 `.pth` 文件。

### 4. 启动聊天页面

```powershell
uv run python webui.py --showcase --device cpu
```

浏览器打开 <http://127.0.0.1:7860>。在“模型选择”中可以切换 SFT、DPO、PPO、GRPO；切换模型后请清空旧对话，再进行公平比较。

要停止程序，回到 PowerShell 窗口按 `Ctrl+C`。关闭网页并不一定会停止后台程序。

## macOS 安装与运行

### 1. 安装 Git 和 uv

打开“终端”，运行：

```bash
xcode-select --install
curl -LsSf https://astral.sh/uv/install.sh | sh
```

按照弹窗完成命令行工具安装，然后关闭终端并重新打开。检查安装：

```bash
git --version
uv --version
```

### 2. 克隆仓库并安装运行环境

```bash
cd ~/Desktop
git clone https://github.com/ziyang02/minimind_from_scratch.git
cd minimind_from_scratch
uv python install 3.12
uv sync --frozen --extra cpu --extra web
```

### 3. 下载并解压模型

从 [GitHub Releases 页面](https://github.com/ziyang02/minimind_from_scratch/releases/latest)下载 `minimind_showcase_models.tar` 和它的 `.sha256` 文件，然后运行：

```bash
cd ~/Downloads
shasum -a 256 -c minimind_showcase_models.tar.sha256
cd ~/Desktop/minimind_from_scratch
tar -xf ~/Downloads/minimind_showcase_models.tar -C .
ls -lh out/showcase
```

校验应显示 `minimind_showcase_models.tar: OK`，最后应列出四个 `.pth` 文件。

### 4. 启动聊天页面

```bash
uv run python webui.py --showcase --device cpu
```

浏览器打开 <http://127.0.0.1:7860>。停止程序时，在终端按 `Control+C`。

## 常见问题

### `git`、`uv` 或 `python` 命令找不到

先关闭终端或 PowerShell 并重新打开。运行项目时应使用 `uv run python`，不依赖系统自己的 `python` 命令。

### 提示找不到 `out/showcase/...pth`

模型包没有解压到仓库根目录。当前目录应该同时包含 `webui.py`、`out/showcase/` 和 `tokenizer_minimind3/`。

### 7860 端口已被使用

换一个端口：

```bash
uv run python webui.py --showcase --device cpu --server-port 7861
```

然后打开 <http://127.0.0.1:7861>。

### 回答很慢或质量不如大型商业模型

这是约 6400 万参数的教学模型，CPU 推理较慢，能力也不能与数十亿参数的商业模型相比。四个 checkpoint 主要用于体验 SFT 与不同后训练方法带来的行为差异。

### 对话内容会上传吗

不会。默认服务只监听本机 `127.0.0.1`，推理在朋友自己的电脑上完成。不要添加 `--share`，也不要把监听地址改为 `0.0.0.0`。

### 如何更新项目

在仓库目录运行：

```bash
git pull
uv sync --frozen --extra cpu --extra web
```

## 仓库所有者发布模型包

Git 会忽略训练权重，所以朋友只运行 `git clone` 得不到模型。仓库所有者需要先在 GitHub 仓库中打开 **Releases → Draft a new release**：

1. 创建标签，例如 `v0.1.0-models`；
2. 上传 `minimind_showcase_models.tar`；
3. 同时上传 `minimind_showcase_models.tar.sha256`；
4. 发布 Release；
5. 再把本指南发给朋友。

不要把接近 1 GB 的模型权重直接提交到普通 Git 历史中。
