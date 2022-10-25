from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import ServerBase
from utils.model_utils import read_data, read_user_data
import numpy as np
# Implementation for FedAvg Server
import time


#  super.init可以继承ServerBase的全部方法，FedAvg是ServerBase的子类
class FedAvg(ServerBase):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all  users
        data = read_data(args.dataset)
        # data是五部分，data[0]clients,依次是groups,train_data,test_data,proxy_data,对于FedAvg，proxy_data为空值
        total_users = len(data[0])
        self.use_adam = 'adam' in self.algorithm.lower()
        # false
        print("Users in total: {}".format(total_users))
        # total_users 20,这是下载的数据里面的人数；arg.num_users 10这是我们在main里面选择的num_users数

        for i in range(total_users):
            id, train_data, test_data = read_user_data(i, data, dataset=args.dataset)
            user = UserAVG(args, id, model, train_data, test_data, use_adam=False)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        print("Number of users / total users:", args.num_users, " / ", total_users)
        print("Finished creating FedAvg server.")
    # FedAvg的模型训练，
    # 1.循环num_glob_iters次训练：随机选择num_users个用户，先对全部user进行evaluate，获得metrics['glob_acc']，metrics['glob_loss']
    #       2.循环num_users个用户：进行训练useravg的train,ClientUpdate函数更新每个user的权重
    #   计算训练um_users的平均时间 metrics['user_train_time']
    #   server进行模型聚合，计算模型聚合的时间 metrics['server_agg_time']

    def train(self, args):
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ", glob_iter, " -------------\n\n")
            self.selected_users = self.select_users(glob_iter, self.num_users)
            # 随机选择num_users个用户
            self.send_parameters(mode=self.mode)
            self.evaluate()     # 对所有20个user进行评估
            self.timestamp = time.time()  # log user-training start time
            for user in self.selected_users:  # allow selected users to train
                user.train(glob_iter, personalized=self.personalized)  # useravg的train
            curr_timestamp = time.time()  # log  user-training end time
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            # train_time是平均给每个user的训练时间
            self.metrics['user_train_time'].append(train_time)
            # Evaluate selected user
            if self.personalized:
                # Evaluate personal model on user for each iteration
                print("Evaluate personal model\n")
                self.evaluate_personalized_model()

            self.timestamp = time.time()  # log server-agg start time
            self.aggregate_parameters()
            curr_timestamp = time.time()  # log  server-agg end time
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)
        #     server聚合模型所用时间
        self.save_results(args)
        self.save_model()
