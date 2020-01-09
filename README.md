
# 简介

Trans 模型在线训练&部署

# 环境要求

- python3
- pytorch
- rabbitmq 3.8
- mysql 5.7

# 使用

1、安装pytorch
- 无gpu：`conda install pytorch torchvision cpuonly -c pytorch`
- 有gpu：https://pytorch.org/get-started/locally/

2、安装其他依赖
```bash
pip install -r requirements.txt
# 或
pip install protobuf grpcio grpcio-tools pika numpy tqdm scikit-learn requests PyMySQL
```

3、编译C++ 文件
```bash
cd shmodel/openke
bash make.sh
```

4、运行
```bash
# 运行服务端（训练模型）
bash bin/train_server.sh
# 运行服务端（部署模型）
bash bin/predict_server.sh
```

# 配置

```bash
cd config
cp config-example.json config-prod.json
```

修改`config/config.json`
```json
{
    "env": "prod"
}
```

修改`config/config-prod.json`
- server.port # 对应[sh4j](https://github.com/Beim/sh4j) 的server.port 配置
- grpc.port # grpc server 监听的端口
- rabbitmq.host
- rabbitmq.port
- rabbitmq.username
- rabbitmq.password
- mysql.host
- mysql.port
- mysql.username
- mysql.password
- gpu # 是否使用gpu

```json
{
    "server": {
        "host": "",
        "port": 18080,
        "protocol": "http"
    },
    "grpc": {
        "port": 8000
    },
    "rabbitmq": {
        "queue_name": "trainJobQueue",
        "host": "localhost",
        "username": "root",
        "password": "123456",
        "port": "5672",
        "durable": true,
        "auto_ack": false,
        "prefetch_count": 1
    },
    "mysql": {
        "host": "localhost",
        "port": 3306,
        "database": "servicehouse",
        "username": "root",
        "password": "123456"
    },
    "path": {
        "trainedmodels": "trainedmodels",
        "benchmarks": "benchmarks/gspace",
        "checkpoint": "checkpoints"
    },
    "update_interval": 5,
    "gpu": false
}
```

# 其他

训练模型使用[OpenKE-PyTorch](https://github.com/thunlp/OpenKE)
```
 @inproceedings{han2018openke,
   title={OpenKE: An Open Toolkit for Knowledge Embedding},
   author={Han, Xu and Cao, Shulin and Lv Xin and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong and Li, Juanzi},
   booktitle={Proceedings of EMNLP},
   year={2018}
 }
```
