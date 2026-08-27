# transduction_sparse.py 安全运行说明（防 "Content Exists Risk" 崩溃）

## 问题（历史背景）
直接运行 `python vision/transduction_sparse.py` 时，脚本 stdout 的中文内容会被放进
tool/result，进而进入下一次模型请求上下文，触发内容审核（"Content Exists Risk
INVALID_REQUEST"），导致对话崩溃（已崩 7-8 轮）。

## 崩溃原理（一句话版）
脚本输出 → tool/result → 对话历史 → 下一次 API 请求携带 → 内容审核层拒绝整个请求
→ 毒内容永久留在历史里 → 之后每次请求都被拒 → 对话"焊死"。读结果也崩，因为
读 = 把内容放进上下文。

## 当前状态：源码级治本已完成（2026-08-28）
`vision\transduction_sparse.py` 已修改：
- 脚本开头把 stdout/stderr 重定向到文件（utf-8）：
  - stdout → `logs\transduction_sparse_out.txt`
  - stderr → `logs\transduction_sparse_err.txt`
- 结尾只向真正的 stdout 打印一行 ASCII 状态：
  `[ok] done - full output: logs/transduction_sparse_out.txt`
  失败时打印 `[err] failed - see logs/transduction_sparse_err.txt`
- 原脚本备份在 `vision\transduction_sparse.py.bak`

**因此现在直接 `python vision/transduction_sparse.py` 也是安全的**——stdout 只有
一行 ASCII 状态，正文全部进文件。任何会话、任何工具里跑都不会再崩。

## 验证记录（全部通过）
- `exit=0`（真实退出码；本机 Start-Process 误报退出码的问题已避开）
- 包装器日志 120 字节 = 纯 ASCII 状态行（UTF-16 BOM 是 PS 5.1 重定向的产物，内容无中文）
- `transduction_sparse_out.txt` = 1807 字节，UTF-8，正文完整
- `transduction_sparse_err.txt` = 0 字节（无错误）
- 源码无 subprocess/os.system/Popen（无其他泄漏通道）

## 用法（包装器保留作保险，也可直接跑）
```
powershell -NoProfile -File .\run_transduction_safe.ps1
# 输出形如：exit=0 bytes=120 lines=1 timeout=False log=logs\transduction_run.log
# 或直接：python vision\transduction_sparse.py   （现在安全）
```

## 看结果
- 脚本完整输出在 `logs\transduction_sparse_out.txt`（人可读，UTF-8）。
- 出错时看 `logs\transduction_sparse_err.txt`。

## 铁律（仍然有效，防旧文件）
以下文件仍含旧的中文输出（"毒"文件，人可读，模型禁读）：
- `logs\exit_capture.txt`、`logs\exit1_diagnosis.txt`、`logs\exit_scan.txt`
- `logs\probe_stdout.txt`、`logs\probe_stderr.txt`
- 项目根目录的 `_transduction_run1*.txt` 等历史输出文件
禁止任何 LLM（包括子代理）read / Get-Content / grep 这些文件的内容。

## 工具脚本（排查留档，可复用）
- `run_transduction_safe.ps1`：包装器（元数据式运行）
- `diagnose_exit1.py` / `scan_exits.py` / `replay_catch.py` / `probe_subprocess.py` /
  `analyze_log_ascii.py`：纯代码排查工具，stdout 只出元数据
