
## Nvidia Driver
[NVIDIA Driver Installation Guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html)


## Terminal Proxy
- set Clash mixed port to 7897
- add below to .zshrc
```bash
export all_proxy=socks5h://127.0.0.1:7897
export http_proxy="http://127.0.0.1:7897"
export https_proxy="http://127.0.0.1:7897"
export no_proxy="localhost,127.0.0.1,::1"
export HTTP_PROXY="http://127.0.0.1:7897"
export HTTPS_PROXY="http://127.0.0.1:7897"
export NO_PROXY="localhost,127.0.0.1,::1"
```

## CMake Ubuntu Apt Repo
https://apt.kitware.com/

## LLVM Ubuntu Apt Repo
https://mirrors.tuna.tsinghua.edu.cn/help/llvm-apt

use /usr/lib/llvm-21 as LLVMINSTALL

## z shell
### Measure initilization time

## Oh my shell
https://ohmyz.sh/

## nvim
### [lazy.nvim](https://lazy.folke.io)


### [coc.nvim](https://github.com/neoclide/coc.nvim)


### [nvim-cmp](https://github.com/hrsh7th/nvim-cmp)

### The vimrc file


### nvchad

## ripgrep
ripgrep recursively searches directories for a regex pattern while respecting your gitignore

https://github.com/BurntSushi/ripgrep/blob/master/GUIDE.md

## fzf
A command-line fuzzy finder

## zoxide
A smarter cd command. Supports all major shells.

