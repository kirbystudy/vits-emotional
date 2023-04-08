chcp 65001
@echo off

:run

:: 标题定义
title sanxingtest

%1 mshta vbscript:CreateObject("Shell.Application").ShellExecute("cmd.exe","/c %~s0 ::","","runas",1)(window.close)&&exit

:: 当前路径
set strPath=%~dp0

:: 下面是检查进程是否存在，不存在启动当前路径下的脚本文件1.bat
netstat -ano|findstr 9999
if %errorlevel%==0 (
	echo Vits服务已存在
) else (
	echo Vits服务不存在，启动
	start  /d"D:\Moe" python JP0.py
  	
)
:: 定时10秒
choice /t 10 /d y /n >nul

goto run

