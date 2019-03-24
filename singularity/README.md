# Singularity 管理员手册指南（旗舰版）

## 快速开始

本小节将介绍如何从零开始，快速构建一个singularity容器镜像。
下文将以构建TensorFlow应用框架为例，依次展开介绍。

**环境要求：**

- 安装singularity（参照第一章的内容）
- 工作目录`share`可用空间： >20GB
- 能够访问公网
- 操作系统：CentOS7

### 第一步 获取系统基础镜像

**说明： 由于`Tensorflow`框架官方在`ubuntu`系统上做过测试验证，所以容器镜像的基础操作系统采用`ubuntu`**

- 编写镜像定义文件 `/share/base.def`

    ```
    Bootstrap: library
    From: ubuntu:18.04
    
    %labels
        Author demo@example.org
        Version v0.0.1
    ```

- 执行构建镜像命令

    ```bash
    singularity build --notest /share/base.sif /share/base.def
    ```

### 第二步 配置Nvidia驱动

⚠️**注意：镜像内安装的Nvidia驱动一定要和超算环境里的nvidia驱动保持一致，否则容器将不能正常运行！**

nvidia驱动官网地址：https://www.nvidia.com/Download/Find.aspx?lang=en-us
选择相应GPU型号和操作系统版本等信息，下载对应版本的驱动，或复制下载链接替换下文`wget https://...` 部分的内容

- 编写镜像定义文件 `/share/base-nvidia.def`
    
    ```
    Bootstrap: localimage
    From: /share/base.sif
    
    %post
        mkdir /nvidia
        cd /nvidia/
        apt-get update
        apt-get install wget -y
        wget https://us.download.nvidia.cn/tesla/410.72/nvidia-diag-driver-local-repo-ubuntu1804-410.72_1.0-1_amd64.deb
        dpkg -i /nvidia/nvidia-diag-driver-local-repo-ubuntu1804-410.72_1.0-1_amd64.deb
        apt-get install gnupg2 -y
        apt-key add /var/nvidia-diag-driver-local-repo-410.72/7fa2af80.pub
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install keyboard-configuration -y
        apt-get install cuda-drivers -y
        rm -rf /nvidia
  ``` 

- 执行构建镜像命令

    ```bash
    singularity build --notest /share/base-nvidia.sif /share/base-nvidia.def
    ```

### 第三步 构建TensorFlow镜像

安装参考 [Tensorflow安装](https://www.tensorflow.org/install/)

- 创建可写的容器目录 `/share/tensorflow`

    ```
    $ sudo singularity build --sandbox /share/tensorflow /share/base-nvidia.sif
    ```

- 进入容器

    ```
    $ sudo singularity shell --writable /share/tensorflow
    ```

    输出如下：
    
    ```
    Singularity tensorflow:~>
    ```
    
- 安装`pip`工具


    ```bash
    Singularity tensorflow:~> apt-get update
    Singularity tensorflow:~> apt-get install python-pip python-dev -y
    ```

- 安装 `tensorflow`

    ```bash
    Singularity tensorflow:~> pip install --user --upgrade tensorflow
    Singularity tensorflow:~> pip install pandas
    ```

- 测试 `tensorflow`

    ```bash
    Singularity tensorflow:~> python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"
    ```

- 创建共享目录 `/Share`

    ⚠️注意： 共享目录名称必须和超算环境的共享目录名称保持一致
    
    ```bash
    Singularity tensorflow:~> mkdir /Share
    ```
    
- 退出容器shell

     ```bash
     Singularity tensorflow:~> exit
     ```
    
- 容器格式转换

    ```bash
    $ sudo singularity build /share/tensorflow.sif /share/tensorflow
    ```

### 第四步 测试实例

- 编写tensorflow测试用例 `/share/tf-hello.py`

    ```python
    #!/usr/bin/env python
    # encoding: utf-8
    import tensorflow as tf
    
    tf.enable_eager_execution()
    tf.add(1, 2).numpy()
    
    hello = tf.constant('Hello, TensorFlow!')
    hello.numpy()
    ```

- 运行测试

    ```
    singularity exec /share/tensorflow.sif python /share/tf-hello.py
    ```

## 第一章 安装

本小节主要介绍CentOS下的安装操作步骤，其它操作系统的安装过程请参阅官网手册[Installation](https://www.sylabs.io/guides/3.0/user-guide/installation.html)

### 1.1 安装依赖环境

```bash
$ sudo yum update -y && \
    sudo yum groupinstall -y 'Development Tools' && \
    sudo yum install -y \
    openssl-devel \
    libuuid-devel \
    libseccomp-devel \
    wget \
    git \
    squashfs-tools
```

### 1.2 安装Go

- 下载`Go`安装包

    下载地址： [https://studygolang.com/dl](https://studygolang.com/dl)
    选择下载版本：go1.11.linux-amd64.tar.gz
    解压软件包： tar -C /usr/local -xzvf go1.11.linux-amd64.tar.gz
    删除软件包：rm go1.11.linux-amd64.tar.gz

- 配置`Go`环境

    ```bash
    $ echo 'export GOPATH=${HOME}/go' >> ~/.bashrc && \
        echo 'export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin' >> ~/.bashrc && \
        source ~/.bashrc
    ```

- 安装`dep`

    ```bash
    $ go get -u github.com/golang/dep/cmd/dep
    ```

### 1.3 下载源代码

```bash
$ go get -d github.com/sylabs/singularity
```

### 1.4 编译安装

```bash
$ export VERSION=v3.0.1  && \
  cd $GOPATH/src/github.com/sylabs/singularity && \
  git fetch && \
  git checkout $VERSION 
```

```bash
$ ./mconfig --prefix=/opt/singularity && \
    make -C ./builddir && \
    sudo make -C ./builddir install
```

```bash
$ . /opt/singularity/etc/bash_completion.d/singularity
```

### 1.5 卸载

```
rm -rf /opt/singularity
```

## 第二章 创建容器

### 2.1 构建容器的几种方法

Singulairty 支持多种方式制作容器。用户可以根据自己的喜好选择合适的方式。
Singularity 使用`build`命令来创建容器。 `build`命令接受一个目标地址作为输入并且生成容器作为输出。
目标地址定义了创建容器的方法，`build` 命令接受的目标地址包括：

- URI若以 *shub://* 开头， 以Singularity Hub (http://singularity-hub.org/)为来源创建容器
- URI若以 *docker://* 开头，以Docker Hub (http://singularity-hub.org/)为来源创建容器
- 本地已经存在的image文件的路径，从已有的image文件创建新的容器
- 本地的目录的路径, 从本地一个sandbox目录创建容器
- Singularity 清单文件路径, 从清单文件创建容器

另外`build`命令可以生成三种格式的image文件。 可以通过相应参数选项来指定输出的image的格式

- 用于生成环境下的压缩只读 *squashfs* 文件系统(默认)
- 用于交互式开发环境下可写 *ext3* file 文件系统(--writable 选项)
- 交互式开发环境下称为sandbox的可写目录结构 *(ch)root directory* (--sandbox 选项)

由于`build`命令接受本地的目标容器文件生成任意格式的新的image文件，所以我们可以将现有的容器文件转换成容易格式的容器文件。

### 2.2 从Container Library构建容器

```bash
$ sudo singularity build lolcow.simg library://sylabs-jms/testing/lolcow
```

- `lolcow.simg`: 构建的容器名称
- `library://sylabs-jms/testing/lolcow`: Container Library URI下载地址
- 默认情况下将创建一个压缩只读的SIF容器，如果想有可写权限可以使用`--sandbox`参数

### 2.3 从Docker Hub构建容器

```bash
$ sudo singularity build lolcow.sif docker://godlovedc/lolcow
```

### 2.4 创建一个可写的容器目录

```bash
$ sudo singularity build --sandbox lolcow/ library://sylabs-jms/testing/lolcow
```

```
$ sudo singularity shell --writable lolcow/
```
### 2.5 容器格式转换

- `development/` 目录转SIF格式文件 `production.sif`

    ```
    $ sudo singularity build production.sif development/
    ```

### 2.6 从singularity定义文件创建容器

- 编写定义文件 `lolcow.def`

```
Bootstrap: docker
From: ubuntu:16.04

%post
    apt-get -y update
    apt-get -y install fortune cowsay lolcat

%environment
    export LC_ALL=C
    export PATH=/usr/games:$PATH

%runscript
    fortune | cowsay | lolcat
```

- 执行创建命令

```bash
$ sudo singularity build lolcow.sif lolcow.def
```

### 2.7 `build` 命令更多选项用法

```bash
$ singularity build --help
```

## 第三章 容器定义文件

singularity容器定义文件简称`def文件`，解释了如何构建自定义的容器。它的内容描述包括了基本的OS或者从开始构建的基本容器，要安装的软件，运行时的环境变量设置，要从主机系统添加的文件以及容器元数据的详细信息。

### 3.1 概述

singularity定义文件分为两部分：

`Header`：标题描述了在容器内构建的核心操作系统。在这里，您将配置容器中所需的基本操作系统功能。您可以指定Linux发行版，特定版本以及必须属于核心安装（从主机系统借用）的软件包。
`Sections`：定义的其余部分由部分组成（有时称为scriptlet或blob数据）。每个部分由一个% 字符定义，后跟特定部分的名称。所有部分都是可选的，def文件可能包含给定部分的多个实例。在构建时执行的部分由/bin/sh解释器执行， 并且可以接受/bin/sh选项。类似地，生成要在运行时执行的脚本的部分可以接受用于的选项/bin/sh
有关def文件的更深入和实用的示例，请参阅[Sylabs示例存储库](https://github.com/sylabs/examples)

### 3.2 Header

`Header`应写在def文件的顶部。它告诉Singularity它应该用来构建容器的基本操作系统。它由几个关键字组成。

每种类型的构建所需的唯一关键字是Bootstrap。它确定 将用于创建要使用的基本操作系统的引导代理程序。例如，library引导代理将从容器库中拉出一个容器作为基础。同样，docker 引导代理会将Docker Hub中的docker层作为基本操作系统从中启动映像。

根据分配给的值Bootstrap，其他关键字也可能在标题中有效。例如，使用library引导代理程序时，From关键字变为有效。请注意以下示例，以便从Container Library构建Debian容器：

```
Bootstrap: library
From: debian:7
```
使用官方镜像安装Centos-7的def文件可能如下所示：

```
Bootstrap: yum
OSVersion: 7
MirrorURL: http://mirror.centos.org/centos-%{OSVERSION}/%{OSVERSION}/os/$basearch/
Include: yum
```

每个引导代理都启用自己的选项和关键字。您可以阅读它们并查看附录中的示例：

- library (images hosted on the Container Library)
- docker (images hosted on Docker Hub)
- shub (images hosted on Singularity Hub)
- localimage (images saved on your machine)
- yum (yum based systems such as CentOS and Scientific Linux)
- debootstrap (apt based systems such as Debian and Ubuntu)
- arch (Arch Linux)
- busybox (BusyBox)
- zypper (zypper based systems such as Suse and OpenSuse)


### 3.3 Sections

bootstrap文件的主要内容分为几个部分。不同的部分在构建过程中的不同时间添加不同的内容或执行命令。请注意，如果任何命令失败，则构建过程将停止。

这是一个使用每个可用部分的示例定义文件。我们将依次讨论每个部分。没有必要在def文件中包含每个部分（或任何部分）。此外，def文件中各部分的顺序并不重要，可以包含同名的多个部分，并且在构建过程中将相互附加。

```
Bootstrap: library
From: ubuntu:18.04

%setup
    touch /file1
    touch ${SINGULARITY_ROOTFS}/file2

%files
    /file1
    /file1 /opt

%environment
    export LISTEN_PORT=12345
    export LC_ALL=C

%post
    apt-get update && apt-get install -y netcat
    NOW=`date`
    echo "export NOW=\"${NOW}\"" >> $SINGULARITY_ENVIRONMENT

%runscript
    echo "Container was created $NOW"
    echo "Arguments received: $*"
    exec echo "$@"

%startscript
    nc -lp $LISTEN_PORT

%test
    grep -q NAME=\"Ubuntu\" /etc/os-release
    if [ $? -eq 0 ]; then
        echo "Container base is Ubuntu as expected."
    else
        echo "Container base is not Ubuntu."
    fi

%labels
    Author d@sylabs.io
    Version v0.0.1

%help
    This is a demo container used to illustrate a def file that uses all
    supported sections.
```

#### `%setup`

%setup安装基本操作系统后，该部分中的命令将在容器外部的主机系统上执行。您可以$SINGULARITY_ROOTFS在%setup部分中使用环境变量引用容器文件系统。

小心这个%setup部分！此scriptlet在主机系统本身的容器外部执行，并使用提升的特权执行。命令%setup可以改变并可能损坏主机。

考虑上面定义文件中的示例：

```
%setup
    touch /file1
    touch ${SINGULARITY_ROOTFS}/file2
```

这里，file1是在主机上的文件系统的根目录下创建的。我们将用于file1演示%files以下部分的用法。它file2是在容器内的文件系统的根目录下创建的。

在Singularity的更高版本中，该%files部分是作为在构建期间将文件从主机系统复制到容器中的更安全的替代方法。由于%setup 在构建期间使用主机系统上的提升权限运行scriptlet 可能存在危险，因此通常不鼓励使用它。

#### `%files`

该%files部分允许您将文件从主机系统复制到容器中，比使用该%setup部分更安全。每行是一个 <source>和<destination>一对，其中，所述源是在主机系统上的路径，并且该目的地是在所述容器的路径。该 <destination> 规范可以被省略，并且将被认为是作为相同的路径 <source>规格。

考虑上面定义文件中的示例：

```
%files
    /file1
    /file1 /opt
```

file1在该%setup 部分期间在主机文件系统的根目录中创建（参见上文）。该%files小脚本将复制file1到容器文件系统的根目录，然后作出的第二个副本file1在容器内/opt。

在执行该%files部分之前复制该部分中的文件，%post以便在构建和配置过程中它们可用。

#### `%environment`

该%environment部分允许您定义将在运行时设置的环境变量。请注意，这些变量在构建时不会通过包含在该%environment部分中而提供。这意味着如果在构建过程中需要相同的变量，则还应在您的%post部分中定义它们。特别：

在构建期间：该%environment部分被写入容器元数据目录中的文件。此文件未来源。
在运行时：源容器元数据目录中的文件来源。
您应该使用在.bashrc或 .profile文件中使用的相同约定。从上面的def文件中考虑这个例子：

```
%environment
    export LISTEN_PORT=12345
    export LC_ALL=C
```

该`$LISTEN_PORT`变量将在`%startscript`下面的部分中使用。该`$LC_ALL`变量对于许多程序（通常用Perl编写）很有用，这些程序在没有设置语言环境时会抱怨。

构建此容器后，可以使用以下命令验证是否在运行时正确设置了环境变量：

```
$ singularity exec my_container.sif env | grep -E 'LISTEN_PORT|LC_ALL'
LISTEN_PORT=12345
LC_ALL=C
```

在构建时生成的变量的特殊情况下，您还可以在该%post部分中将环境变量添加到容器中（参见下文）。

在构建时，该%environment部分的内容将写入/.singularity.d/env/90-environment.sh容器内部调用的文件中。$SINGULARITY_ENVIRONMENT在%post（见下文）中重定向到变量的文本被添加到一个名为的文件中/.singularity.d/env/91-environment.sh。

在运行时，脚本/.singularity/env按顺序获取。这意味着该%post部分中的变量优先于添加的 变量%environment。

有关Singularity容器环境的更多信息，请参阅环境和元数据。

#### `%post`

%post在构建时安装基本OS之后，该部分中的命令在容器内执行。在这里，你会从网上下载文件与工具，如git和wget，安装新的软件和库，编写配置文件，创建新的目录，等等。

考虑上面定义文件中的示例：
```
%post
    apt-get update && apt-get install -y netcat
    NOW=`date`
    echo "export NOW=\"${NOW}\"" >> $SINGULARITY_ENVIRONMENT
```

此%postscriptlet使用Ubuntu包管理器apt更新容器并安装程序netcat（将在%startscript下面的 部分中使用）。

该脚本还在构建时设置环境变量。请注意，无法预期此变量的值，因此在该%environment部分期间无法设置。对于这种情况， $SINGULARITY_ENVIRONMENT提供变量。将文本重定向到此变量将导致将其写入/.singularity.d/env/91-environment.sh将在运行时获取的调用文件 。请注意，设置的变量%post优先于上述%environment部分中设置的 变量。

#### `%runscript`

该%runscript部分的内容被写入容器内的文件，该文件在运行容器映像时执行（通过 命令或直接作为命令执行容器）。调用容器时，容器名称后面的参数将传递给运行脚本。这意味着您可以（并且应该）处理运行脚本中的参数。singularity run

考虑上面def文件中的示例：

```
%runscript
    echo "Container was created $NOW"
    echo "Arguments received: $*"
    exec echo "$@"
```

在此脚本中，创建容器的时间通过`$NOW`变量（在`%post`上面的部分中设置）进行回显。在运行时传递给容器的选项打印为单个字符串（`$*`），然后通过引用的数组（`$@`）传递给echo ，确保所执行的命令正确解析所有参数。在`exec`之前的最后`echo`命令替换在过程表中的当前条目（其原本是调用奇异）。因此，运行脚本shell进程不再存在，只剩下容器内运行的进程。

运行使用此def文件构建的容器将产生以下结果：

```
$ ./my_container.sif
Container was created Thu Dec  6 20:01:56 UTC 2018
Arguments received:

$ ./my_container.sif this that and the other
Container was created Thu Dec  6 20:01:56 UTC 2018
Arguments received: this that and the other
this that and the other
```

#### `%startscript`

与该%runscript部分类似，该部分的内容%startscript 在构建时写入容器内的文件。发出命令时执行此文件。instance start

考虑上面def文件中的示例。

```
%startscript
    nc -lp $LISTEN_PORT
```

这里netcat程序用于监听$LISTEN_PORT变量指示的端口上的TCP流量（在%environment上面的部分中设置）。可以像这样调用脚本：

```
$ singularity instance start my_container.sif instance1
INFO:    instance started successfully

$ lsof | grep LISTEN
nc        19061               vagrant    3u     IPv4             107409      0t0        TCP *:12345 (LISTEN)

$ singularity instance stop instance1
Stopping instance1 instance of /home/vagrant/my_container.sif (PID=19035)
```

#### `%test`

该%test部分在构建过程的最后运行，以使用您选择的方法验证容器。您还可以使用该test命令通过容器本身执行此scriptlet 。

考虑上面def文件中的示例：

```
%test
    grep -q NAME=\"Ubuntu\" /etc/os-release
    if [ $? -eq 0 ]; then
        echo "Container base is Ubuntu as expected."
    else
        echo "Container base is not Ubuntu."
    fi
```

这个（有点傻）脚本测试基本操作系统是否是Ubuntu。您还可以编写一个脚本来测试二进制文件是否已正确下载和构建，或者该软件在自定义硬件上按预期工作。如果要在不运行该%test部分的情况下构建容器（例如，如果构建系统没有将在生产系统上使用的相同硬件），则可以使用--notest构建选项执行此操作：

```bash
$ sudo singularity build --notest my_container.sif my_container.def
```
在使用此def文件构建的容器上运行test命令会产生以下结果：

```bash
$ singularity test my_container.sif
Container base is Ubuntu as expected.
```

#### `%labels`

该%labels部分用于向/.singularity.d/labels.json容器中的文件添加元数据 。通用格式是名称 - 值对。

考虑上面def文件中的示例：

```
%labels
    Author d@sylabs.io
    Version v0.0.1
```

查看标签的最简单方法是检查图像：

```
$ singularity inspect my_container.sif

{
    "Author": "d@sylabs.io",
    "Version": "v0.0.1",
    "org.label-schema.build-date": "Thursday_6_December_2018_20:1:56_UTC",
    "org.label-schema.schema-version": "1.0",
    "org.label-schema.usage": "/.singularity.d/runscript.help",
    "org.label-schema.usage.singularity.deffile.bootstrap": "library",
    "org.label-schema.usage.singularity.deffile.from": "ubuntu:18.04",
    "org.label-schema.usage.singularity.runscript.help": "/.singularity.d/runscript.help",
    "org.label-schema.usage.singularity.version": "3.0.1"
}
```
从构建过程中自动捕获的一些标签。您可以在此处详细了解标签和元数据。

#### `%help`

%help在构建期间，该部分中的任何文本都会转录到容器中的元数据文件中。然后可以使用该run-help命令显示该文本 。

考虑上面def文件中的示例：

```
%help
    This is a demo container used to illustrate a def file that uses all
    supported sections.
```

构建完成后，可以这样显示：

```
$ singularity run-help my_container.sif
    This is a demo container used to illustrate a def file that uses all
    supported sections.
```

### 3.4 Apps

在某些情况下，为每个应用程序构建不同的容器可能是多余的，具有几乎相同的依赖性。Singularity支持基于标准容器集成格式（SCI-F）的概念在内部模块中安装应用程序

以下脚本演示了如何使用SCI-F模块将2个不同的应用程序构建到同一个容器中：

```
Bootstrap: docker
From: ubuntu

%environment
    GLOBAL=variables
    AVAILABLE="to all apps"

##############################
# foo
##############################

%apprun foo
    exec echo "RUNNING FOO"

%applabels foo
   BESTAPP FOO

%appinstall foo
   touch foo.exec

%appenv foo
    SOFTWARE=foo
    export SOFTWARE

%apphelp foo
    This is the help for foo.

%appfiles foo
   foo.txt

##############################
# bar
##############################

%apphelp bar
    This is the help for bar.

%applabels bar
   BESTAPP BAR

%appinstall bar
    touch bar.exec

%appenv bar
    SOFTWARE=bar
    export SOFTWARE
```

一个`%appinstall`部分相当于`%post`一个特定的应用程序。同样，`%appenv`等同于应用程序版本%environment等等。

所述`%app*`部分可以沿着任何主部分（即存在 `%post`，`%runscript`，`%environment`等等）。与其他部分一样，部分的顺序`%app*`并不重要。

使用这些%app*部分将应用程序安装到模块后，该--app 选项可用，允许以下功能：

要在容器中运行特定的应用程序：

```
% singularity run --app foo my_container.sif
RUNNING FOO
```

`$SOFTWARE`在上面的def文件中为两个应用程序定义了相同的环境变量。您可以执行以下命令来搜索活动环境变量列表，并`grep`根据我们指定的应用程序确定变量是否更改：

```
$ singularity exec --app foo my_container.sif env | grep SOFTWARE
SOFTWARE=foo

$ singularity exec --app bar my_container.sif env | grep SOFTWARE
SOFTWARE=bar
```

### 3.5 构建容器的最佳实践

制作容器时，最好考虑以下事项：

- 总包，程序，数据和文件安装到操作系统中的位置（例如，不`/home`，`/tmp`或者可能在得到普遍绑定任何其他目录）。
- 记录您的容器。如果您的脚本没有提供帮助，请写一个 `%help`或`%apphelp`一节。一个好的容器告诉用户如何与它进行交互。
- 如果需要定义任何特殊环境变量，请将它们添加到构建配方的`%environment`和`%appenv`部分。
- 文件应始终由系统帐户拥有（UID小于500）。
- 确保这样的敏感文件`/etc/passwd`，`/etc/group`和 `/etc/shadow`不包含的秘密。
- 从定义文件而不是手动更改的沙箱构建生产容器。这确保了最大可重复性并减轻了“黑匣子”效应。

## 第四章 绑定路径和挂载

如果由系统管理员启用，Singularity允​​许您使用绑定挂载将主机系统上的目录映射到容器中的目录。这使您可以轻松地在主机系统上读写数据。

当Singularity“交换”容器内的主机操作系统时，主机文件系统将无法访问。但您可能希望从容器内读取和写入主机系统上的文件。要启用此功能，Singularity将通过两种主要方法将目录绑定回容器：系统定义的绑定路径和用户定义的绑定路径。

### 4.1 绑定系统定义的路径

系统管理员可以定义每个容器内自动包含的绑定路径。一些绑定路径是自动派生的（例如用户的主目录），一些是静态定义的（例如，Singularity配置文件中的绑定路径）。在默认配置中，目录$HOME，/tmp，/proc，/sys， /dev，和$PWD是系统定义的绑定路径中。

### 4.2 绑定用户定义的路径

如果系统管理员已启用用户对绑定的控制，您将能够在容器中请求自己的绑定路径。

奇异动作命令（run，exec，shell，和 将接受命令行选项指定绑定的路径，并且也将履行的（或 ）环境变量。该选项的参数是绑定路径规范的格式的逗号分隔的字符串 ，where 和是分别在容器外部和内部的路径。如果没有给出，则设置为等于 。可以将挂载选项（）指定为（只读）或 （读/写，这是默认值）。可以多次指定选项，也可以使用逗号分隔的绑定路径规范字符串。`instance start--bind/-B$SINGULARITY_BIND$SINGULARITY_BINDPATHsrc[:dest[:opts]]srcdestdestsrcoptsrorw--bind/-B`

#### 4.2.1 指定绑定路径

以下是在容器中使用主机上的`--bind`选项和绑定的示例（不需要已经存在于容器中）：`/data` `/mnt` `/mnt`

```bash
$ ls /data
bar  foo

$ singularity exec --bind /data:/mnt my_container.sif ls /mnt
bar  foo
```

您可以使用以下语法在单个命令中绑定多个目录：

```bash
$ singularity shell --bind /opt,/data:/mnt my_container.sif
```

这将绑定`/opt`在`/opt`容器`/data` 中的主机上`/mnt`以及容器中的主机上。

使用环境变量而不是命令行参数，这将是：

```shell
$ export SINGULARITY_BIND="/opt,/data:/mnt"

$ singularity shell my_container.sif
```

使用环境变量`$SINGULARITY_BIND`，即使将容器作为带有运行脚本的可执行文件运行，也可以绑定路径。如果将许多目录绑定到Singularity容器中并且它们不会更改，您甚至可以通过在`.bashrc` 文件中设置此变量来获益。

#### 4.2.2 使用的注意事项 `--bind` 与 `--writable` 标志

要在容器内安装绑定路径，必须在容器中定义绑定点。绑定点是容器内的目录，Singularity可以将其用作绑定主机系统上的目录的目标。

从版本3.0开始，Singularity将尽力将挂载请求的路径绑定到容器中，而不管容器中是否存在适当的绑定点。即使没有“叠加fs”功能，奇点通常也可以执行此操作。

但是，当与 `--writable` 标志结合使用时，绑定到容器内不存在点的路径可能会导致意外行为，因此不允许这样做。如果需要结合 `--writable` 标志指定绑定路径 ，请确保容器中存在适当的绑定点。如果它们尚不存在，则需要修改容器并创建它们。

## 第五章 持续叠加

持久覆盖目录允许您将可写文件系统覆盖在不可变的只读容器上，以实现读写访问的错觉。

概述
持久性叠加层是一个“位于压缩的，不可变的SIF容器顶部”的目录。安装新软件或创建和修改文件时，overlay目录会存储更改。

如果要将SIF容器用作可写入的，则可以创建一个目录以用作持久性叠加。然后，您可以指定要在运行时使用该`--overlay`选项作为覆盖使用该目录。

您可以使用以下命令的持久叠加：

- run
- exec
- shell
- instance.start

### 5.1 用法

要使用持久性叠加层，必须先拥有容器。

```shell
$ sudo singularity build ubuntu.sif library://ubuntu
```

然后，您必须创建一个目录。（您也可以将该--overlay选项与旧的可写ext3映像一起使用。）

```shell
$ mkdir my_overlay
```

现在，您可以将此overlay目录与容器一起使用。请注意，必须是root才能使用overlay目录。

```shell
$ sudo singularity shell --overlay my_overlay/ ubuntu.sif

Singularity ubuntu.sif:~> touch /foo

Singularity ubuntu.sif:~> apt-get update && apt-get install -y vim

Singularity ubuntu.sif:~> which vim
/usr/bin/vim

Singularity ubuntu.sif:~> exit
```

您将发现您的更改在会话中保持不变，就像您使用可写容器一样。

```shell
$ sudo singularity shell --overlay my_overlay/ ubuntu.sif

Singularity ubuntu.sif:~> ls /foo
/foo

Singularity ubuntu.sif:~> which vim
/usr/bin/vim

Singularity ubuntu.sif:~> exit
```

如果在没有--overlay目录的情况下装载容器，则更改将会消失。

```shell
$ sudo singularity shell ubuntu.sif

Singularity ubuntu.sif:~> ls /foo
ls: cannot access 'foo': No such file or directory

Singularity ubuntu.sif:~> which vim

Singularity ubuntu.sif:~> exit
```

## 第六章 环境和元数据

Singularity容器支持在构建过程中可以添加到容器的环境变量和标签。如果要在构建期间查找环境变量以在主机系统上设置环境，请参阅构建环境部分。

### 6.1 概述

通过在定义文件中添加环境变量，可以将它们包含在容器中：

在%environment定义文件的部分中。

```
Bootstrap: library
From: library/alpine

%environment
    VARIABLE_ONE = hello
    VARIABLE_TWO = world
    export VARIABLE_ONE VARIABLE_TWO
```

或者在%post定义文件的部分中。

```
Bootstrap: library
From: library/alpine

%post
    echo 'export VARIABLE_NAME=variable_value' >>$SINGULARITY_ENVIRONMENT
```

您还可以使用以下%labels部分为容器添加标签：

```
Bootstrap: library
From: library/alpine

%labels
    OWNER = Joana    
```

要查看容器中的标签，请使用以下inspect命令：

```
$  singularity inspect mysifimage.sif
```

这将为您提供以下输出：

```json
{
    "OWNER": "Joana",
    "org.label-schema.build-date": "Monday_07_January_2019_0:01:50_CET",
    "org.label-schema.schema-version": "1.0",
    "org.label-schema.usage": "/.singularity.d/runscript.help",
    "org.label-schema.usage.singularity.deffile.bootstrap": "library",
    "org.label-schema.usage.singularity.deffile.from": "debian:9",
    "org.label-schema.usage.singularity.runscript.help": "/.singularity.d/runscript.help",
    "org.label-schema.usage.singularity.version": "3.0.1-236.g2453fdfe"
}
```

默认情况下会创建许多这些标签，但您也可以看到上面示例中添加的自定义标签。

该inspect命令具有其他选项，可用于查看容器的元数据。

### 6.2 环境变量

如果您从Container Library或Docker Hub构建容器，则构建时将在容器中包含该环境。您还可以在定义文件中定义新的环境变量，如下所示：

```
Bootstrap: library
From: library/alpine

%environment
    #First define the variables
    VARIABLE_PATH=/usr/local/bootstrap
    VARIABLE_VERSION=3.0
    #Then export them
    export VARIABLE_PATH VARIABLE_VERSION
```
    
您可能需要在该%post 部分期间将环境变量添加到容器中。例如，在安装某些软件之前，您可能不知道变量的适当值。要在期间向环境添加变量，%post可以使用$SINGULARITY_ENVIRONMENT 具有以下语法的变量：

```
%post
    echo 'export VARIABLE_NAME=variable_value' >>$SINGULARITY_ENVIRONMENT
```
    
该%environment部分中的文本将附加到文件， `/.singularity.d/env/90-environment.sh`而重定向到的文本`$SINGULARITY_ENVIRONMENT`将显示在文件中 `/.singularity.d/env/91-environment.sh`。如果未`$SINGULARITY_ENVIRONMENT`在该`%post`部分中重定向到任何 内容，则该文件 `/.singularity.d/env/91-environment.sh`将不存在。

因为文件/.singularity.d/env来源是按字母顺序排列的，所以使用$SINGULARITY_ENVIRONMENT优先级添加的变量优先于通过该%environment部分添加的变量。

如果需要在运行时在容器中定义变量，则在执行Singularity时会传递一个前缀为的变量SINGULARITYENV_。这些变量将自动转置，前缀将被剥离。例如，假设我们要将变量设置HELLO为有值world。我们可以这样做：

```
$ SINGULARITYENV_HELLO=world singularity exec centos7.img env | grep HELLO
HELLO=world
```

该--cleanenv选项可用于删除主机环境并使用最小环境执行容器。

```shell
$ singularity exec --cleanenv centos7.img env
LD_LIBRARY_PATH=:/usr/local/lib:/usr/local/lib64
SINGULARITY_NAME=test.img
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PWD=/home/gmk/git/singularity
LANG=en_US.UTF-8
SHLVL=0
SINGULARITY_INIT=1
SINGULARITY_CONTAINER=test.img
```

如果没有该--cleanenv标志，主机系统上的环境将在运行时出现在容器中。

如果您需要$PATH在运行时更改容器，可以使用一些特殊的环境变量：

`SINGULARITYENV_PREPEND_PATH=/good/stuff/at/beginning` 将目录添加到开头的 `$PATH`

`SINGULARITYENV_APPEND_PATH=/good/stuff/at/end` 将目录附加到结尾` $PATH
SINGULARITYENV_PATH=/a/new/path`覆盖`$PATH`容器内

### 6.3 标签

您的容器存储有关其构建的元数据，Docker标签以及您在%labels部分构建期间定义的自定义标签。

对于使用Singularity 3.0及更高版本生成的容器，使用rc1 Label Schema表示标签。例如：


```bash
$ singularity inspect jupyter.sif
```

```json
    {
        "OWNER": "Joana",
        "org.label-schema.build-date": "Friday_21_December_2018_0:49:50_CET",
        "org.label-schema.schema-version": "1.0",
        "org.label-schema.usage": "/.singularity.d/runscript.help",
        "org.label-schema.usage.singularity.deffile.bootstrap": "library",
        "org.label-schema.usage.singularity.deffile.from": "debian:9",
        "org.label-schema.usage.singularity.runscript.help": "/.singularity.d/runscript.help",
        "org.label-schema.usage.singularity.version": "3.0.1-236.g2453fdfe"
    }
```

您会注意到一个标签不属于标签架构`OWNER`。这是用户在引导期间提供的标签。

您可以在引导程序文件中向容器添加自定义标签：

```
Bootstrap: docker
From: ubuntu: latest

%labels
  OWNER Joana
```

`inspect`命令对于查看标签和其他容器元数据非常有用。下一节将详细介绍其各种选项。

### 6.4 inspect命令

inspect命令使您能够使用定义文件打印出添加到容器中的标签和/或其他元数据。

#### `--labels`

此标志对应于`inspect`命令的默认行为。当你运行`singularity inspect <your-container.sif>`时，你会得到这样的输出。

```
$ singularity inspect --labels jupyter.sif

```

```json
{
    "org.label-schema.build-date": "Friday_21_December_2018_0:49:50_CET",
    "org.label-schema.schema-version": "1.0",
    "org.label-schema.usage": "/.singularity.d/runscript.help",
    "org.label-schema.usage.singularity.deffile.bootstrap": "library",
    "org.label-schema.usage.singularity.deffile.from": "debian:9",
    "org.label-schema.usage.singularity.runscript.help": "/.singularity.d/runscript.help",
    "org.label-schema.usage.singularity.version": "3.0.1-236.g2453fdfe"
}
```

这跟跑步一样。singularity inspect jupyter.sif

#### `--deffile`

此标志为您提供用于创建容器的def文件。

```
$ singularity inspect --deffile jupyter.sif
```

输出看起来像：

```
Bootstrap: library
From: debian:9

%help
    Container with Anaconda 2 (Conda 4.5.11 Canary) and Jupyter Notebook 5.6.0 for Debian 9.x (Stretch).
    This installation is based on Python 2.7.15

%environment
    JUP_PORT=8888
    JUP_IPNAME=localhost
    export JUP_PORT JUP_IPNAME

%startscript
    PORT=""
    if [ -n "$JUP_PORT" ]; then
    PORT="--port=${JUP_PORT}"
    fi

    IPNAME=""
    if [ -n "$JUP_IPNAME" ]; then
    IPNAME="--ip=${JUP_IPNAME}"
    fi

    exec jupyter notebook --allow-root ${PORT} ${IPNAME}

%setup
    #Create the .condarc file where the environments/channels from conda are specified, these are pulled with preference to root
    cd /
    touch .condarc

%post
    echo 'export RANDOM=123456' >>$SINGULARITY_ENVIRONMENT
    #Installing all dependencies
    apt-get update && apt-get -y upgrade
    apt-get -y install \
    build-essential \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git
    rm -rf /var/lib/apt/lists/*
    apt-get clean
    #Installing Anaconda 2 and Conda 4.5.11
    wget -c https://repo.continuum.io/archive/Anaconda2-5.3.0-Linux-x86_64.sh
    /bin/bash Anaconda2-5.3.0-Linux-x86_64.sh -bfp /usr/local
    #Conda configuration of channels from .condarc file
    conda config --file /.condarc --add channels defaults
    conda config --file /.condarc --add channels conda-forge
    conda update conda
    #List installed environments
    conda list
```

这是jupyter.sif容器的定义文件。

#### `--runscript`

此标志显示图像的运行脚本。

```bash
$ singularity inspect --runscript jupyter.sif
```

输出看起来像：

```bash
#!/bin/sh
OCI_ENTRYPOINT=""
OCI_CMD="bash"
# ENTRYPOINT only - run entrypoint plus args
if [ -z "$OCI_CMD" ] && [ -n "$OCI_ENTRYPOINT" ]; then
SINGULARITY_OCI_RUN="${OCI_ENTRYPOINT} $@"
fi

# CMD only - run CMD or override with args
if [ -n "$OCI_CMD" ] && [ -z "$OCI_ENTRYPOINT" ]; then
if [ $# -gt 0 ]; then
    SINGULARITY_OCI_RUN="$@"
else
    SINGULARITY_OCI_RUN="${OCI_CMD}"
fi
fi

# ENTRYPOINT and CMD - run ENTRYPOINT with CMD as default args
# override with user provided args
if [ $# -gt 0 ]; then
    SINGULARITY_OCI_RUN="${OCI_ENTRYPOINT} $@"
else
    SINGULARITY_OCI_RUN="${OCI_ENTRYPOINT} ${OCI_CMD}"
fi

exec $SINGULARITY_OCI_RUN
```

#### `--test`

此标志显示图像的测试脚本。

```shell
$ singularity inspect --test jupyter.sif
```

这将从`%test`定义文件中输出相应的部分。

#### `--environment`

此标志显示图像的环境设置。在设置的各个环境变量`%environment`部分（因此，在那些 `90-environment.sh`）和`SINGULARITY_ENV`在运行时设置（即位于`91-environment.sh`）变量将被打印出来。

```
$ singularity inspect --environment jupyter.sif
```

输出看起来像：

```
==90-environment.sh==
#!/bin/sh

JUP_PORT=8888
JUP_IPNAME=localhost
export JUP_PORT JUP_IPNAME

==91-environment.sh==
export RANDOM=123456
```

正如你所看到的，`JUP_PORT`而`JUP_IPNAME`在以前定义`%environment`把定义文件的部分，同时显示关于使用随机变量`SINGULARITYENV_`的变量，所以在这种情况下， `SINGULARITYENV_RANDOM`变量设置和运行时输出。

#### `--helpfile`

此标志将在`%help`其定义文件的部分中显示容器的描述。

你可以这样称呼它：

```shell
$ singularity inspect --helpfile jupyter.sif
```

输出看起来像：

```
Container with Anaconda 2 (Conda 4.5.11 Canary) and Jupyter Notebook 5.6.0 for Debian 9.x (Stretch).
This installation is based on Python 2.7.15
```

#### `--json`

此标志使您可以以`JSON`格式输出标签。

你可以这样称呼它：

```
$ singularity inspect --json jupyter.sif
```

输出看起来像：

```json
{
         "attributes": {
                 "labels": "{\n\t\"org.label-schema.build-date\": \"Friday_21_December_2018_0:49:50_CET\",\n\t\"org.label-schema.schema-version\": \"1.0\",\n\t\"org.label-schema.usage\": \"/.singularity.d/runscript.help\",\n\t\"org.label-schema.usage.singularity.deffile.bootstrap\": \"library\",\n\t\"org.label-schema.usage.singularity.deffile.from\": \"debian:9\",\n\t\"org.label-schema.usage.singularity.runscript.help\": \"/.singularity.d/runscript.help\",\n\t\"org.label-schema.usage.singularity.version\": \"3.0.1-236.g2453fdfe\"\n}"
         },
         "type": "container"
}
```

### 6.5 容器元数据

在容器内部，元数据存储在`/.singularity.d`目录中。您可能不应该直接编辑这些文件，但了解它们的位置和作用可能会有所帮助：

```
/.singularity.d/

├── actions
│   ├── exec
│   ├── run
│   ├── shell
│   ├── start
│   └── test
├── env
│   ├── 01-base.sh
|   ├── 10-docker2singularity.sh
│   ├── 90-environment.sh
│   ├── 91-environment.sh
|   ├── 94-appsbase.sh
│   ├── 95-apps.sh
│   └── 99-base.sh
├── labels.json
├── libs
├── runscript
├── runscript.help
├── Singularity
└── startscript
```

- **actions**：此目录包含帮助程序脚本以允许容器执行操作命令。（例如`exec`，`run`或`shell`）在Singularity的更高版本中，这些文件可以在运行时动态写入。
- **env**：启动容器时，此目录中的所有* .sh文件都以字母数字顺序提供。对于遗留目的，有一个称为`/environment`指向 的符号链接`/.singularity.d/env/90-environment.sh`。
- **labels.json**：存储上述容器标签的json文件。
- **libs**：在运行时，用户可以请求将一些主机系统库映射到容器中（`--nv`例如，使用选项）。如果是这样，这是他们的目的地。
- **runscript**：当使用该`run`命令调用容器或将其称为可执行文件时，将执行此文件中的命令。出于传统目的，有一个称为`/singularity`指向此文件的符号链接。
- **runscript.help**：包含该`%help` 部分中添加的描述。
- **singularity**：这是用于生成容器的定义文件。如果使用多个定义文件来生成容器，则其他Singularity文件将以数字顺序显示在名为的子目录中`bootstrap_history`。
- **startscript**：当使用该命令调用容器时，将执行此文件中的命令。`instance start`


## 第七章 签名和验证

Singularity 3.0引入了创建和管理PGP密钥的能力，并使用它们来签署和验证容器。这为Singularity用户共享容器提供了一种可信方法。它确保了作者所希望的原始容器的逐位复制。

### 7.1 从容器库验证容器

`verify`命令将允许您验证是否已使用PGP密钥对容器进行了签名。要将此功能与从容器库中提取的图像一起使用，必须首先为Sylabs Cloud生成访问令牌。如果您还没有有效的访问令牌，请按照下列步骤操作：

1. 请访问：`https：//cloud.sylabs.io/`
2. 单击“登录Sylabs”并按照登录步骤操作。
3. 单击您的登录ID（与登录一个相同和更新的按钮）。
4. 从下拉菜单中选择“访问令牌”。
5. 点击“帐户管理”页面中的“管理我的API令牌”按钮。
6. 单击“创建”。
7. 单击“新API令牌”页面中的“将令牌复制到剪贴板”。
8. 将令牌字符串粘贴到您的`~/.singularity/sylabs-token`文件中。

 现在，您可以验证从库中提取的容器，确保它们是原始图像的逐位复制。

```
$ singularity pull library://alpine

$ singularity verify alpine_latest.sif
Verifying image: alpine_latest.sif
Data integrity checked, authentic and signed by:
    Sylabs Admin <support@sylabs.io>, KeyID 51BE5020C508C7E9
```

在此示例中，您可以看到Sylabs Admin已对容器进行了签名。

### 7.2 签署自己的容器

#### 7.2.1 生成和管理PGP密钥

要签署自己的容器，首先需要生成一个或多个密钥。

如果您在生成任何密钥之前尝试对容器进行签名，Singularity将引导您完成创建新密钥的交互过程。或者您可以`newpair`在`key`命令组中使用子命令，如下所示：

```
$ singularity keys newpair
Enter your name (e.g., John Doe) : Dave Godlove
Enter your email address (e.g., john.doe@example.com) : d@sylabs.io
Enter optional comment (e.g., development keys) : demo
Generating Entity and OpenPGP Key Pair... Done
Enter encryption passphrase :
```

该`list`子命令将显示所有已创建或保存locally.的关键

```
$ singularity keys list
Public key listing (/home/david/.singularity/sypgp/pgp-public):

0) U: Dave Godlove (demo) <d@sylabs.io>
   C: 2018-10-08 15:25:30 -0400 EDT
   F: 135E426D67D8416DE1D6AC7FFED5BBA38EE0DC4A
   L: 4096
   --------
```

在上面的输出中，字母代表以下内容：


- U：用户
- C：创建日期和时间
- F：指纹
- L：密钥长度


生成密钥后，您可以选择 使用指纹将其推送到[Keystore](https://cloud.sylabs.io/keystore)，如下所示：

```shell
$ singularity keys push 135E426D67D8416DE1D6AC7FFED5BBA38EE0DC4A
public key `135E426D67D8416DE1D6AC7FFED5BBA38EE0DC4A` pushed to server successfully
```

这将允许其他人验证您已签名的图像。

如果删除本地公共PGP密钥，则可以随时查找并再次下载。

```shell
$ singularity keys search Godlove
Search results for 'Godlove'

Type bits/keyID     Date       User ID
--------------------------------------------------------------------------------
pub  4096R/8EE0DC4A 2018-10-08 Dave Godlove (demo) <d@sylabs.io>
--------------------------------------------------------------------------------

$ singularity keys pull 8EE0DC4A
1 key(s) fetched and stored in local cache /home/david/.singularity/sypgp/pgp-public
```

但请注意，这只是恢复公共密钥（用于验证）到本地计算机，并不会恢复私有密钥（用于签名）。

签名并验证自己的容器
现在您已生成密钥，您可以使用它来对图像进行签名，如下所示：

```shell
$ singularity sign my_container.sif
Signing image: my_container.sif
Enter key passphrase:
Signature created and applied to my_container.sif
```

由于您的公共PGP密钥已保存在本地，因此您无需联系密钥库即可验证映像。

```shell
$ singularity verify my_container.sif
Verifying image: my_container.sif
Data integrity checked, authentic and signed by:
    Dave Godlove (demo) <d@sylabs.io>, KeyID FED5BBA38EE0DC4A
```

如果您已将密钥推送到密钥库，则还可以在没有本地密钥的情况下验证此图像。为了演示这一点，首先删除本地密钥，然后再次尝试使用`verify`命令。

```shell
$ rm ~/.singularity/sypgp/*

$ singularity verify my_container.sif
Verifying image: my_container.sif
INFO:    key missing, searching key server for KeyID: FED5BBA38EE0DC4A...
INFO:    key retreived successfully!
Store new public key 135E426D67D8416DE1D6AC7FFED5BBA38EE0DC4A? [Y/n] y
Data integrity checked, authentic and signed by:
    Dave Godlove (demo) <d@sylabs.io>, KeyID FED5BBA38EE0DC4A
```

在交互式提示符处回答“是”将在本地存储公钥，这样您下次验证容器时就不必再次联系密钥库。

##  第八章 安全

Singularity 3.0为容器运行时引入了许多与安全相关的新选项。本文档将描述用户在运行Singularity容器时指定安全范围和上下文的新方法。

### 8.1 Linux功能

Singularity完全支持在用户或组的基础上授予和撤销Linux功能。例如，让我们假设管理员决定授予用户打开原始套接字的功能，以便他们可以`ping`在容器中使用 ，通过功能控制二进制文件（即最新版本的CentOS）。

为此，管理员将发出如下命令：

```shell
$ sudo singularity capability add --user david CAP_NET_RAW
```

这意味着用户`david`刚被授予权限（通过Linux功能）在Singularity容器中打开原始套接字。

管理员可以使用该 命令检查此更改是否有效。`capability list`

```shell
$ sudo singularity capability list --user david
CAP_NET_RAW
```

要利用此新功能，用户`david`还必须在执行带有`--add-caps`标志的容器时请求该功能，如下所示：

```shell
$ singularity exec --add-caps CAP_NET_RAW library://centos ping -c 1 8.8.8.8
PING 8.8.8.8 (8.8.8.8) 56(84) bytes of data.
64 bytes from 8.8.8.8: icmp_seq=1 ttl=128 time=18.3 ms

--- 8.8.8.8 ping statistics ---
1 packets transmitted, 1 received, 0% packet loss, time 0ms
rtt min/avg/max/mdev = 18.320/18.320/18.320/0.000 ms
```

如果管理员决定不再需要允许用户`dave` 在Singularity容器中打开原始套接字，他们可以撤销相应的Linux功能，如下所示：

```
$ sudo singularity capability drop --user david CAP_NET_RAW
```

在和子也将接受不区分大小写的关键字授予或撤销所有的Linux功能，用户或组。同样，该选项将接受该关键字。当然，使用此关键字时应该谨慎行事。capabiltiy adddropall--add-capsall

### 8.2 安全相关的操作选项

Singularity 3.0引入了许多可以传递给action命令的新标志; `shell`，`exec`并`run`允许细粒度的安全控制。

#### `--add-caps`

如上所述，`--add-caps`在启动容器时将“激活”Linux功能，前提是管理员使用该命令授予用户这些功能。此选项还将接受不区分大小写的关键字，以添加管理员授予的每个功能。`capability add` `all`

#### `--allow-setuid`
SetUID位允许程序作为拥有二进制文件的用户执行。最着名的SetUID二进制文件由root拥有，允许用户使用提升的权限执行命令。但是其他SetUID二进制文件可能允许用户将命令作为服务帐户执行。

默认情况下，Singularity容器中不允许使用SetUID作为安全预防措施。但root用户可以覆盖此预防措施，并允许SetUID二进制文件在Singularity容器中按预期运行，其 `--allow-setuid`选项如下：

```
$ sudo singularity shell --allow-setuid some_container.sif
```

#### `--keep-privs`
管理员可以通过将文件中的参数设置为或 分别设置不同的默认功能集或将root用户的默认功能减少为零。如果此更改生效，则root用户可以使用该选项覆盖该文件并输入具有完整功能的容器。
root default capabilitiessingularity.conffilenosingularity.conf--keep-privs

```shell
$ sudo singularity exec --keep-privs library://centos ping -c 1 8.8.8.8
PING 8.8.8.8 (8.8.8.8) 56(84) bytes of data.
64 bytes from 8.8.8.8: icmp_seq=1 ttl=128 time=18.8 ms

--- 8.8.8.8 ping statistics ---
1 packets transmitted, 1 received, 0% packet loss, time 0ms
rtt min/avg/max/mdev = 18.838/18.838/18.838/0.000 ms
```

#### `--drop-caps`

默认情况下，root用户在进入容器时具有一整套功能。当您以root用户身份启动容器时，可以选择删除特定功能以增强安全性。

例如，要删除root用户在容器内打开原始套接字的功能：

```shell
$ sudo singularity exec --drop-caps CAP_NET_RAW library://centos ping -c 1 8.8.8.8
ping: socket: Operation not permitted
```

该`drop-caps`选项还将接受不区分大小写的关键字`all` 作为在进入容器时删除所有功能的选项。

#### `--security`

`--security`标志允许root用户在Singularity容器中利用SELinux，AppArmor和seccomp等安全模块。您还可以在运行时更改容器内用户的UID和GID。

例如：

```
$ sudo whoami
root

$ sudo singularity exec --security uid:1000 my_container.sif whoami
david
```

要使用seccomp将命令列入黑名单，请遵循以下过程。（从安全角度来看，实际上最好使用白名单命令，但这对于一个简单的示例就足够了。）请注意，此示例是在Ubuntu上运行的，而Singularity是使用libseccomp-dev和pkg-config 包作为依赖项安装的。

首先编写配置文件。使用Singularity安装示例配置文件，通常位于/usr/local/etc/singularity/seccomp-profiles/default.json。对于此示例，我们将使用更简单的配置文件将mkdir命令列入黑名单 。

```json
{
    "defaultAction": "SCMP_ACT_ALLOW",
    "archMap": [
        {
            "architecture": "SCMP_ARCH_X86_64",
            "subArchitectures": [
                "SCMP_ARCH_X86",
                "SCMP_ARCH_X32"
            ]
        }
    ],
    "syscalls": [
        {
            "names": [
                "mkdir"
            ],
            "action": "SCMP_ACT_KILL",
            "args": [],
            "comment": "",
            "includes": {},
            "excludes": {}
        }
    ]
}
```

我们将保存文件/home/david/no_mkdir.json。然后我们可以像这样调用容器：

```
$ sudo singularity shell --security seccomp:/home/david/no_mkdir.json my_container.sif

Singularity> mkdir /tmp/foo
Bad system call (core dumped)
```
请注意，尝试使用列入黑名单的mkdir命令会导致核心转储。

`--security`选项接受的完整参数列表如下：

```
--security="seccomp:/usr/local/etc/singularity/seccomp-profiles/default.json"
--security="apparmor:/usr/bin/man"
--security="selinux:context"
--security="uid:1000"
--security="gid:1000"
--security="gid:1000:1:0" (multiple gids, first is always the primary group)
```

## 第九章 网络

Singularity 3.0引入了与cni的完全集成 ，以及一些使网络虚拟化变得容易的新功能。

为了方便这些功能，动作命令（exec，run和shell）中添加了一些新选项，该--net选项也已更新。这些选项只能由root用户使用。

### `--dns`
该--dns选项允许您指定要添加到/etc/resolv.conf文件的以逗号分隔的DNS服务器列表。

```shell
$ nslookup sylabs.io | grep Server
Server:             127.0.0.53

$ sudo singularity exec --dns 8.8.8.8 ubuntu.sif nslookup sylabs.io | grep Server
Server:             8.8.8.8

$ sudo singularity exec --dns 8.8.8.8 ubuntu.sif cat /etc/resolv.conf
nameserver 8.8.8.8
```

### `--hostname`

该--hostname选项接受字符串参数以更改容器中的主机名。

```shell
$ hostname
ubuntu-bionic

$ sudo singularity exec --hostname hal-9000 my_container.sif hostname
hal-9000
```

### `--net`

传递该--net标志将导致容器在启动时加入新的网络命名空间。Singularity 3.0中的新功能，默认情况下也会设置桥接接口。

```shell
$ hostname -I
10.0.2.15

$ sudo singularity exec --net my_container.sif hostname -I
10.22.0.4
```

### `--network`

该--network选项只能与--net 标志一起调用。它接受以逗号分隔的网络类型字符串。每个条目都会在容器内部显示一个专用接口。

```shell
$ hostname -I
172.16.107.251 10.22.0.1

$ sudo singularity exec --net --network ptp ubuntu.sif hostname -I
10.23.0.6

$ sudo singularity exec --net --network bridge,ptp ubuntu.sif hostname -I
10.22.0.14 10.23.0.7
```

在调用时，该--network选项在奇点配置目录（通常/usr/local/etc/singularity/network/）中搜索与所请求的网络类型相对应的cni配置文件。默认情况下，几个配置文件与Singularity一起安装，对应于以下网络类型：

- bridge
- PTP
- ipvlan
- macvlan

管理员还可以定义自定义网络配置并将它们放在同一目录中，以使用户受益。

### `--network-args`

`--network-args`选项提供了一种方便的方法来指定直接传递给cni插件的参数。它必须与 --net旗帜一起使用。

例如，假设您要 在容器内的端口80上启动NGINX服务器，但是您希望将其映射到容器外部的端口8080：

```shell
$ sudo singularity instance start --writable-tmpfs \
    --net --network-args "portmap=8080:80/tcp" docker://nginx web2
```

上面的命令将启动在名为的后台实例中运行的Docker Hub官方NGINX映像web2。NGINX实例需要能够写入磁盘，因此我们使用该--writable-tmpfs参数在内存中分配一些空间。--net使用该--network-args选项时，该标志是必需的 ，并指定portmap=8080:80/tcp将容器内的端口80映射到主机上的8080的参数。

现在我们可以在容器内启动NGINX：

```
$ sudo singularity exec instance://web2 nginx
```

`curl`命令可用于验证NGINX是否按预期在主机端口8080上运行。

```
$ curl localhost:8080
10.22.0.1 - - [16/Oct/2018:09:34:25 -0400] "GET / HTTP/1.1" 200 612 "-" "curl/7.58.0" "-"
<!DOCTYPE html>
<html>
<head>
<title>Welcome to nginx!</title>
<style>
    body {
        width: 35em;
        margin: 0 auto;
        font-family: Tahoma, Verdana, Arial, sans-serif;
    }
</style>
</head>
<body>
<h1>Welcome to nginx!</h1>
<p>If you see this page, the nginx web server is successfully installed and
working. Further configuration is required.</p>

<p>For online documentation and support please refer to
<a href="http://nginx.org/">nginx.org</a>.<br/>
Commercial support is available at
<a href="http://nginx.com/">nginx.com</a>.</p>

<p><em>Thank you for using nginx.</em></p>
</body>
</html>
```

有关cni的更多信息，请查看 cni规范。

## 第十章 Cgroup支持

从Singularity 3.0开始，用户可以使用cgroup限制容器资源。

### 10.1 概述

可以通过TOML文件配置和使用奇点cgroups支持。通常安装示例文件 /usr/local/etc/singularity/cgroups/cgroups.toml。您可以复制和编辑此文件以满足您的需要。然后，当您需要限制容器资源时，通过使用路径作为--apply-cgroups选项的参数来应用TOML文件中的设置， 如下所示：

```
$ sudo singularity shell --apply-cgroups /path/to/cgroups.toml my_container.sif
```

该`--apply-cgroups`选项只能与`root`权限一起使用。

### 10.2 示例

#### 10.2.1 限制内存

若要将容器使用的内存量限制为500MB（524288000字节），请按照此示例操作。首先，创建一个cgroups.toml这样的文件并将其保存在您的主目录中。

```
[memory]
    limit = 524288000
```

像这样启动你的容器：

```
$ sudo singularity instance start --apply-cgroups /home/$USER/cgroups.toml \
    my_container.sif instance1
```

之后，您可以验证容器是否仅使用500MB内存。（此示例假定这instance1是唯一正在运行的实例。）

```
$ cat /sys/fs/cgroup/memory/singularity/*/memory.limit_in_bytes
524288000
```

完成此示例后，请确保使用以下命令清理实例。

```
$ sudo singularity instance stop instance1
```

类似地，可以通过启动实例并检查相应子目录的内容来测试其余示例/sys/fs/cgroup/。

#### 10.2.2 限制CPU

使用以下策略之一限制CPU资源。cpu配置文件的部分可以使用以下内容限制内存：

##### shares

这相当于与具有cpu份额的其他cgroup的比率。通常默认值是1024。这意味着如果您想允许使用单个CPU的50％，您将设置512为值。

```
[cpu]
    shares = 512
```

如果系统中有足够的空闲CPU周期，由于调度程序的工作保留性，cgroup可以获得超过其CPU份额，因此包含的进程可以消耗所有CPU周期，即使比率为50％。仅当两个或多个进程与其CPU周期需求冲突时才应用该比率。

##### quota/period

您可以对cgroup可以使用的CPU周期强制执行硬限制，因此包含的进程的使用量不能超过为cgroup设置的CPU时间。quota允许您配置cgroup每个时段可以使用的CPU时间量。默认值为100毫秒（100000us）。因此，如果要在100ms的时间段内将CPU时间限制为20ms：

```
[cpu]
    period = 100000
    quota = 20000
```

##### cpus/mems 

您还可以使用以下cpus/mems字段限制对特定CPU和关联内存节点的访问：

```
[cpu]
    cpus = "0-1"
    mems = "0-1"
```

容器对CPU 0和CPU 1的访问权限有限。

注意

为cpus和两者设置相同的值很重要mems。

有关使用cgroup限制CPU的详细信息，请参阅以下外部链接：

- 红帽资源管理指南第3.2节CPU
- 红帽资源管理指南第3.4节CPUSET
- 内核调度程序文档

#### 10.2.3 限制IO

您可以限制和监视对块设备的I / O访问。使用[blockIO]配置文件的 部分执行此操作，如下所示：

```
[blockIO]
    weight = 1000
    leafWeight = 1000
``` 
   
weight并leafWeight接受10和之间的值1000。

weight 是所有设备上组的默认权重，直到并且除非被每个设备规则覆盖。

leafWeight 与权重有关，目的是决定在与cgroup的子cgroup竞争时给定cgroup中的任务有多重。

要覆盖weight/leafWeight的/dev/loop0和/dev/loop1块设备，你会做这样的事情：

```
[blockIO]
    [[blockIO.weightDevice]]
        major = 7
        minor = 0
        weight = 100
        leafWeight = 50
    [[blockIO.weightDevice]]
        major = 7
        minor = 1
        weight = 100
        leafWeight = 50   
```

您可以/dev/loop0 使用以下配置将块读取/写入速率限制为每秒16MB 。速率以每秒字节数指定。

```
[blockIO]
    [[blockIO.throttleReadBpsDevice]]
        major = 7
        minor = 0
        rate = 16777216
    [[blockIO.throttleWriteBpsDevice]]
        major = 7
        minor = 0
        rate = 16777216
```

要将块读/写速率限制为/dev/loop0 块设备上的每秒1000 IO（IOPS），可以执行以下操作。速率在IOPS中指定。

```
[blockIO]
    [[blockIO.throttleReadIOPSDevice]]
        major = 7
        minor = 0
        rate = 1000
    [[blockIO.throttleWriteIOPSDevice]]
        major = 7
        minor = 0
        rate = 1000
        
```

有关限制IO的更多信息，请参阅以下外部链接：

- 红帽资源管理指南3.1 blkio
- 内核块IO控制器文档
- 内核CFQ调度程序文档

#### 10.2.4 限制设备访问
您可以限制读取，写入或创建设备。在此示例中，容器配置为仅能够读取或写入`/dev/null`。

```
[[devices]]
    access = "rwm"
    allow = false
[[devices]]
    access = "rw"
    allow = true
    major = 1
    minor = 3
    type = "c"
```

有关限制设备访问的更多信息，请参阅Red Hat资源管理指南第3.5节“设备”。

## 第十一章 参考资料

- https://www.sylabs.io/guides/3.0/user-guide
- https://www.sylabs.io/guides/3.0/admin-guide
- https://hub.docker.com/
- https://cloud.sylabs.io/library
- https://github.com/sylabs/singularity
- https://github.com/sylabs/examples
- https://www.nvidia.com/download/driverResults.aspx/140708/en-us
- https://github.com/ufoym/deepo
- https://github.com/tensorflow/tensorflow
- https://www.tensorflow.org/install/pip?lang=python2


