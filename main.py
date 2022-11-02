#!/usr/bin/env python
import argparse
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverFedProx import FedProx
from FLAlgorithms.servers.serverFedDistill import FedDistill
from FLAlgorithms.servers.serverFedGen import FedGen
from FLAlgorithms.servers.serverFedEnsemble import FedEnsemble
from utils.model_utils import create_model
from utils.plot_utils import *
import torch
from multiprocessing import Pool


def create_server_n_user(args, i):
    model = create_model(args.model, args.dataset, args.algorithm)
    # model是cnn,dataset是Mnist-alpha0.1-ratio0.1，EMnist-alpha0.1-ratio0.1，algorithm无用
    if 'FedAvg' in args.algorithm:
        server = FedAvg(args, model, i)
    elif 'FedGen' in args.algorithm:
        server = FedGen(args, model, i)
    elif 'FedProx' in args.algorithm:
        server = FedProx(args, model, i)
    elif 'FedDistill' in args.algorithm:
        server = FedDistill(args, model, i)
    elif 'FedEnsemble' in args.algorithm:
        server = FedEnsemble(args, model, i)
    else:
        print("Algorithm {} has not been implemented.".format(args.algorithm))
        exit()
    return server


def run_job(args, i):
    torch.manual_seed(i)
    # 设置随机种子，如果i相同则每次rand生成的结果都一样，便于复现。比如说后面使用了三次rand函数，三个随机数可能互不相同，但是一定是同样大小同样顺序的三个数
    print("\n\n         [ Start training iteration {} ]           \n\n".format(i))
    # Generate model
    server = create_server_n_user(args, i)
    if args.train:
        server.train(args)  # 调用各自算法的train和test，如果是FedAvg，则调用Class FedAvg的train函数，即serveravg.py
        server.test()


def main(args):
    for i in range(args.times):
        run_job(args, i)
    print("Finished training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Mnist-alpha0.1-ratio0.5")   # 使用的数据集，需要先使用指令下载
    parser.add_argument("--model", type=str, default="cnn")     # 只有cnn，如果用MLP效果不如cnn
    parser.add_argument("--train", type=int, default=1, choices=[0, 1])  # 这个不用管，是1才能开始训练
    parser.add_argument("--algorithm", type=str, default="FedGen")  # 可以输入的是，FedAvg，FedProx，FedDistill，FedEnsemble，FedGen
    parser.add_argument("--batch_size", type=int, default=32)   # 一次训练所抓取的数据样本数量
    parser.add_argument("--gen_batch_size", type=int, default=32, help='number of samples from generator')  # 生成器一次所抓取的样本数量
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--personal_learning_rate", type=float, default=0.01,
                        help="Personalized learning rate to calculate theta approximately using K steps")
    parser.add_argument("--ensemble_lr", type=float, default=1e-4, help="Ensemble learning rate")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=1, help="Regularization term")     # 正则化
    parser.add_argument("--mix_lambda", type=float, default=0.1, help="Mix lambda for FedMXI baseline")
    parser.add_argument("--embedding", type=int, default=0, help="Use embedding layer in generator network")
    parser.add_argument("--num_glob_iters", type=int, default=10)   # 训练的次数，外循环，在serverFedGen里面
    parser.add_argument("--local_epochs", type=int, default=5)      # 对选中的每个user，进行local_epochs次训练，在userFedGen里面
    parser.add_argument("--num_users", type=int, default=10, help="Number of Users per round")  # 选中的用户数量
    parser.add_argument("--K", type=int, default=1, help="Computation steps")
    parser.add_argument("--times", type=int, default=3, help="running time")  # 运行次数，进行times次的训练和测试
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="run device (cpu | cuda)")  # 可选cpu和cuda
    parser.add_argument("--result_path", type=str, default="results", help="directory path to save results")    # 结果输出路径

    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Ensemble learing rate       : {}".format(args.ensemble_lr))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.num_users))
    print("Number of global rounds       : {}".format(args.num_glob_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("Device            : {}".format(args.device))
    print("=" * 80)
    main(args)
