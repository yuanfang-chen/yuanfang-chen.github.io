## xx
man brew

## Glossary
- formula: 包定义（从上游源码构建）
- cask: 包定义（从上游release包取）
- prefix: Homebrew的安装根目录 , e.g. /opt/homebrew or /home/linuxbrew/.linuxbrew
- keg: 包的某一个版本的安装目录, e.g. /opt/homebrew/Cellar/foo/0.1
- rack: 包的所有版本的根目录, e.g. /opt/homebrew/Cellar/foo
- keg-only: 未软连接到prefix的包
- opt prefix: 包的active版本的软连接, e.g. /opt/homebrew/opt/foo
- Cellar: racks的目录, e.g. /opt/homebrew/Cellar
- Caskroom: casks的目录, e.g. /opt/homebrew/Caskroom
- external command: Homebrew源码中没有的brew子命令，指第三方扩展的子命令
- tap: formulae, casks或者external command的目录或者git仓
- bottle: 放在Cellar中的rack上的预构建的keg (相对于从上游源码构建的keg)

## [external command](https://docs.brew.sh/External-Commands)

