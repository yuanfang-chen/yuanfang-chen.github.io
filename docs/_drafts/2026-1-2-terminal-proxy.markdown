
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

