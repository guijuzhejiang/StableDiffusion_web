# coding=utf-8
# @Time : 2023/3/3 上午10:26
# @File : startup.py
import sys
import os


if __name__ == "__main__":
    # os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['ACCELERATE'] = 'False'
    from modules.shared import cmd_opts
    cmd_opts.listen = True
    cmd_opts.enable_insecure_extension_access = True
    cmd_opts.xformers = True
    # nowebui = True
    nowebui = False
    import webui
    print(f"Launching {'API server' if '--nowebui' in sys.argv else 'Web UI'} with arguments: {' '.join(sys.argv[1:])}")
    if nowebui:
        webui.api_only()
    else:
        webui.webui()
