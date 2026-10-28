# 安装与启动

## 环境

- Python 3.9 或更高版本，包含 pip、venv 和 sqlite3。
- 桌面浏览器，例如 Chrome 或 Edge。
- 程序目录具有写入权限。
- 首次安装可连接 Python 包源，下载 PyYAML 6.0.3。

程序使用本地 CPU 计算。安装完成后可离线使用，无需 GPU 或模型权重。

## macOS

首次使用：

1. 双击 `install.command`。
2. 等待出现 `Environment and configuration check passed.`。
3. 双击 `start.command`。
4. 浏览器打开 <http://127.0.0.1:8001>。

之后直接运行 `start.command`。使用期间保持终端窗口打开，按 `Control+C` 停止服务。

## macOS / Linux 终端

将示例路径替换为程序目录：

```bash
cd "/path/to/PCS-WebUI-20261027"
sh install.sh
sh start.sh
```

安装脚本在程序目录创建 `.venv/` 并安装依赖。日常使用只需执行 `sh start.sh`。

Ubuntu/Debian 如果提示缺少 venv 或 ensurepip，先安装对应 Python 版本的 venv 软件包，再运行安装脚本。

## Windows

首次使用：

1. 安装 Python，确认 `py` 或 `python` 命令可用。
2. 双击 `install.bat`，等待安装完成。
3. 双击 `start.bat`。
4. 浏览器打开 <http://127.0.0.1:8001>。

之后直接运行 `start.bat`。保持命令窗口打开，按 `Ctrl+C` 停止服务。

PowerShell 命令：

```powershell
cd "C:\path\to\PCS-WebUI-20261027"
py -3 install.py
.\.venv\Scripts\python.exe start.py
```

没有 `py` 时，使用 `python install.py`。

## 端口和环境检查

macOS / Linux：

```bash
# 只检查依赖和模型配置
sh start.sh --check

# 使用 8010 端口
sh start.sh --port 8010
```

Windows：

```powershell
.\.venv\Scripts\python.exe start.py --check
.\.venv\Scripts\python.exe start.py --port 8010
```

更换端口后访问 <http://127.0.0.1:8010>。

## 局域网访问

在服务所在电脑上运行：

```bash
sh start.sh --host 0.0.0.0 --port 8001
```

Windows 使用：

```powershell
.\.venv\Scripts\python.exe start.py --host 0.0.0.0 --port 8001
```

其他电脑通过服务机器的局域网 IP 访问，例如 `http://192.168.1.10:8001`。默认启动只允许本机访问。

服务没有登录和用户隔离。访问者共用实验库，同一时刻执行一个计算任务。

## 使用已有 Python 环境

在程序目录执行：

```bash
python3 -m pip install -r requirements.txt
python3 -B dashboard_v2/server.py --host 127.0.0.1 --port 8001
```

Windows 将 `python3` 换为 `py -3` 或 `python`。

`pip install .` 也可安装核心包及依赖。页面和 YAML 仍从程序目录读取，运行时需保留 `dashboard_v2/`、`ssu_sim/` 和 `configs/`。

## 离线安装

先在操作系统、CPU 架构和 Python 版本相同的联网电脑上下载依赖：

```bash
python3 -m pip download --only-binary=:all: -r requirements.txt -d wheels
```

将 `wheels/` 放到离线电脑的程序目录中。目标电脑需已安装 Python 和 venv。

macOS / Linux：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links wheels -r requirements.txt
sh start.sh
```

Windows：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-index --find-links wheels -r requirements.txt
.\.venv\Scripts\python.exe start.py
```

## 移动目录和备份

移动程序目录前停止服务。移动后重新创建 `.venv/`：另存或移除旧 `.venv/`，再运行安装脚本。

实验数据位于 `dashboard_v2/.data/`。备份前停止服务，复制整个目录，包括可能存在的 WAL/SHM 文件。重装 Python 环境不会清空实验数据。

## 常见问题

| 问题 | 处理方法 |
| --- | --- |
| 找不到 python3、python 或 py | 安装 Python，重新打开终端确认命令可用 |
| macOS 双击脚本无法启动 | 打开终端，进入程序目录，执行 `sh install.sh` 和 `sh start.sh` |
| PyYAML 下载失败 | 检查网络、代理和 Python 包源设置，或使用离线安装 |
| 提示尚未安装 | 在当前程序目录重新运行安装脚本 |
| Address already in use | 停止之前的服务，或用 `--port` 指定其他端口 |
| 浏览器打不开 | 查看启动窗口中的报错，确认服务在运行，并核对地址和端口 |
| YAML 或模型配置报错 | 运行 `--check`，根据错误信息检查对应配置文件 |
| HTTP 409 / another task is running | 等待当前计算完成，或在页面取消后重新提交 |
| 实验保存失败 | 检查 `.data/` 目录或自定义数据库路径的写入权限 |
| 导出按钮禁用 | 输入已改变，重新计算或恢复上一轮输入 |
| Not evaluated | 查看 Case 详情中的原因；该场景的部分计算尚未实现 |

页面操作和参数说明见 [README.md](README.md)。
