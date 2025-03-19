# 1 megatron-lm 环境搭建

## 1.1 验证CUDA版本
nvcc --version  # 应显示 11.x 版本
nvidia-smi      # 确认驱动版本支持CUDA版本

## 1.2 安装基础编译工具
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build

## 1.3 安装python虚拟环境
conda create -n megatron python=3.10 -y
conda activate megatron

## 1.4 安装PyTorch (需与CUDA版本对应)
最新版本，带cuda

## 1.5 安装完整版cuda (带driver的安装)
- [下载地址](https://developer.nvidia.com/cuda-downloads)

*注意：可以选择之前的版本号* <br>

```shell
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

## 1.6 安装cudnn

- [下载地址](https://developer.nvidia.com/cudnn-downloads?target_os=Linux)

## 1.7 TransformerEngine 安装

- [安装指导](https://github.com/NVIDIA/TransformerEngine)


# 2 安装Megatron-LM
## 2.1 开发模式
```shell
pip install -e .
python setup.py develop
pip install --no-cache-dir -e /root/projects/Megatron-LM
```

*不保存下载的包到本地缓存目录。不重用之前缓存的包文件，强制从远程仓库重新下载。* <br>

| 特性               | `python setup.py build`         | `pip install -e`              |
|--------------------|---------------------------------|-------------------------------|
| **安装位置**       | 不安装                          | 安装到当前环境                |
| **代码修改同步**   | 需重新安装                      | 实时生效                      |
| **依赖管理**       | 不处理依赖                      | 自动安装依赖项                |
| **C扩展构建**      | 显式构建                        | 自动触发构建                  |
| **虚拟环境兼容性** | 需手动激活环境                  | 自动识别当前环境              |
| **现代Python推荐** | 不推荐                          | 推荐                        |

# 2.1 生产环境

```shell
# 直接安装（推荐）
pip install .

# 或分步操作
python setup.py build
python setup.py install
```
