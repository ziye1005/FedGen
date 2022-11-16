from FLAlgorithms.users.userFedDKDGen import UserFedDKDGen
from FLAlgorithms.servers.serverbase import ServerBase
from utils.model_utils import read_data, read_user_data, read_user_data2, aggregate_user_data, create_generative_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import os
import copy
import time

MIN_SAMPLES_PER_LABEL = 1


class FedDKDGen(ServerBase):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all users
        data = read_data(args.dataset)
        # data是五部分，data[0]clients,依次是groups,train_data,test_data,proxy_data,对于FedAvg，proxy_data为空值
        total_users = len(data[0])      # total_users表示所下载的数据集中user数量
        self.total_test_samples = 0
        self.local = 'local' in self.algorithm.lower()
        self.use_adam = 'adam' in self.algorithm.lower()
        print("Users in total: {}".format(total_users))
        # total_users 20,这是下载的数据里面的人数；arg.num_users 10这是我们在main里面选择的num_users数
        # 不同
        self.early_stop = 20  # stop using generated samples after 20 local epochs 在20个本地epoch之后停止使用生成的样本
        self.student_model = copy.deepcopy(self.model)
        # 生成产生式模型，用于广播给选中的user
        self.generative_model = create_generative_model(args.dataset, args.algorithm, self.model_name, args.embedding)
        if not args.train:
            print('number of generator parameteres: [{}]'.format(self.generative_model.get_number_of_parameters()))
            print('number of model parameteres: [{}]'.format(self.model.get_number_of_parameters()))
        self.latent_layer_idx = self.generative_model.latent_layer_idx
        # 用于包含蒸馏的算法，FedDistill,FedEnsemble,FedGen，初始化一些ensemble_lr，以及generative_alpha这些，并输出
        self.init_ensemble_configs()
        print("latent_layer_idx: {}".format(self.latent_layer_idx))
        print("label embedding {}".format(self.generative_model.embedding))
        print("ensemeble learning rate: {}".format(self.ensemble_lr))
        print("ensemeble alpha = {}, beta = {}, eta = {}".format(self.ensemble_alpha, self.ensemble_beta,
                                                                 self.ensemble_eta))
        print("generator alpha = {}, beta = {}".format(self.generative_alpha, self.generative_beta))
        # 初始化三种损失，使用NLLloss、KL散度（相对熵） KLDivLoss、交叉熵损失
        self.init_loss_fn()
        # 得到available_labels 该数据集下的全部标签
        self.train_data_loader, self.train_iter, self.available_labels = aggregate_user_data(data, args.dataset,
                                                                                             self.ensemble_batch_size)
        # 更新generative_optimizer 和 generative_lr_scheduler
        # torch.optim.Adam是optimizer调用Adam优化算法， 需要先构造一个优化器对象optimizer用于保存当前状态
        # Adam本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。它的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。
        # params(iterable)：可用于迭代优化的参数或者定义参数组的dicts；
        # lr (float, optional) ：学习率(默认: 1e-3)，在这里是1e-4；
        # betas (Tuple[float, float], optional)：用于计算梯度的平均和平方的系数(默认: (0.9, 0.999))；
        # eps (float, optional)：为了提高数值稳定性而添加到分母的一个项(默认: 1e-8)；
        # weight_decay (float, optional)：权重衰减(如L2惩罚)(默认: 0)，在这里1e-2；
        # amsgrad 是否使用AMSGrad的变体,默认false。

        #  torch.optim.lr_scheduler提供了一些根据epoch调整学习率的方法，一般是随着epoch的增大而逐渐减小学习率
        # ExponentialLR表示每个epoch都做一次更新，gamma是更新lr的乘法因子，有序调整————指数衰减调整(Exponential)
        self.generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98)
        # 更新optimizer 和 lr_scheduler，公式与前一组相同
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.98)
        # 不同结束
        # 创建total_users 20 个用户，avg也有这个步骤
        self.users = []
        for i in range(total_users):
            id, train_data, test_data, y_train, label_info = read_user_data2(i, data, dataset=args.dataset, count_labels=True)
            # count_labels=True Gen统计标签分布，但是avg不统计
            self.total_train_samples += len(train_data)
            self.total_test_samples += len(test_data)
            # 这句话没用，藏起来
            # id, train, test = read_user_data(i, data, dataset=args.dataset)
            # 将generative model Gw(self.generative_model)，全局模型预测层theta（self.latent_layer_idx）初始值-1，
            # 全局标签分布（label_info），全部标签信息（available_labels）这增加的四项一起送去UserFedGen取初始化user
            user = UserFedDKDGen(
                args, id, model, self.generative_model,
                train_data, test_data,
                self.available_labels, self.latent_layer_idx, label_info,
                use_adam=self.use_adam, y_train=y_train)
            self.users.append(user)
        print("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        print("Data from {} users in total.".format(total_users))
        print("Finished creating FedDistillGen server.")

    # serverFedGen.train仅有两处不同：chosen_verbose_user以及派生的user.train的调用、训练生成器train_generator
    # FedGen的模型训练，
    # 1.循环num_glob_iters次训练：随机选择num_users个用户，先对全部user进行evaluate，获得metrics['glob_acc']，metrics['glob_loss']
    #   选择一个0，19的随机数作为index即user_id作为chosen_verbose_user
    #       2.循环num_users个用户：判断chosen_verbose_user与当前user_id是否一致，作为增加的参数送给userFedGen.train进行训练
    #   计算训练num_users的平均时间 metrics['user_train_time']
    #   训练生成器train_generator
    #   server进行模型聚合，计算模型聚合的时间 metrics['server_agg_time']
    def train(self, args):
        # pretraining
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number FedDKD: ", glob_iter, " -------------\n\n")
            # return_idx=True表示调用serverbase.select_users返回user_idxs是数组
            self.selected_users, self.user_idxs = self.select_users(glob_iter, self.num_users, return_idx=True)
            if not self.local:
                self.send_parameters(mode=self.mode)  # broadcast averaged prediction model
            self.evaluate(glob_iter=glob_iter)     # 调用serverbase.evaluate
            # 这一行不同
            chosen_verbose_user = np.random.randint(0, len(self.users))     # 选择一个0，19的随机数作为index即user_id
            self.timestamp = time.time()  # log user-training start time
            for user_id, user in zip(self.user_idxs, self.selected_users):  # allow selected users to train
                # 这一行不同
                verbose = user_id == chosen_verbose_user    # 表示是否相同的boolean型，是否打印
                # perform regularization using generated samples after the first communication round
                user.train(     # 调用userFedGen.train
                    glob_iter,
                    personalized=self.personalized,
                    early_stop=self.early_stop,
                    verbose=verbose and glob_iter > 0,
                    regularization=glob_iter > 0)   # 第一轮glob_iter所有user的regularization是false；后面轮次全是true
            curr_timestamp = time.time()  # log  user-training end time
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)
            if self.personalized:
                # Evaluate personal model on user for each iteration
                print("Evaluate personal model\n")
                self.evaluate_personalized_model()

            self.timestamp = time.time()  # log server-agg start time
            # 不同，多了 训练生成器
            self.train_generator(
                self.batch_size,
                epoches=self.ensemble_epochs // self.n_teacher_iters,
                latent_layer_idx=self.latent_layer_idx,
                verbose=True
            )
            # server参数聚合，调用serverbase.aggregate_parameters
            self.aggregate_parameters()
            curr_timestamp = time.time()
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)
            if glob_iter > 0 and glob_iter % 20 == 0 and self.latent_layer_idx == 0:
                self.visualize_images(self.generative_model, glob_iter, repeats=10)

        self.save_results(args)
        self.save_model()

    def train_generator(self, batch_size, epoches=1, latent_layer_idx=-1, verbose=False):
        # 学习一个生成器，可以在给定标签y下找到一个共识的潜在表示
        # Learn a generator that find a consensus latent representation z, given a label 'y'.
        # :param batch_size:
        # :param epoches:
        # :param latent_layer_idx: 如果设置为-1(-2)，获取最后一层(或倒数第二层)的潜在表示。
        # :param verbose: 打印损失信息
        # :return: Do not return anything.
        # self.generative_regularizer.train()
        self.label_weights, self.qualified_labels = self.get_label_weights()
        TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0, 0

        def update_generator_(n_iters, student_model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS):
            self.generative_model.train()
            # 测试集上评估性能
            student_model.eval()
            for i in range(n_iters):
                self.generative_optimizer.zero_grad()
                y = np.random.choice(self.qualified_labels, batch_size)
                y_input = torch.LongTensor(y)
                # feed to generator，前向预测
                gen_result = self.generative_model(y_input, latent_layer_idx=latent_layer_idx, verbose=True)
                # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
                gen_output, eps = gen_result['output'], gen_result['eps']
                # get losses decoded = self.generative_regularizer(gen_output) regularization_loss = beta *
                # self.generative_model.dist_loss(decoded, eps) # map generated z back to eps
                diversity_loss = self.generative_model.diversity_loss(eps, gen_output)  # encourage different outputs

                # get teacher loss
                teacher_loss = 0
                teacher_logit = 0
                for user_idx, user in enumerate(self.selected_users):
                    user.model.eval()
                    weight = self.label_weights[y][:, user_idx].reshape(-1, 1)
                    expand_weight = np.tile(weight, (1, self.unique_labels))
                    user_result_given_gen = user.model(gen_output, start_layer_idx=latent_layer_idx, logit=True)
                    user_output_logp_ = F.log_softmax(user_result_given_gen['logit'], dim=1)
                    teacher_loss_ = torch.mean( \
                        self.generative_model.crossentropy_loss(user_output_logp_, y_input) * \
                        torch.tensor(weight, dtype=torch.float32))
                    teacher_loss += teacher_loss_
                    teacher_logit += user_result_given_gen['logit'] * torch.tensor(expand_weight, dtype=torch.float32)

                # get student loss
                student_output = student_model(gen_output, start_layer_idx=latent_layer_idx, logit=True)
                student_loss = F.kl_div(F.log_softmax(student_output['logit'], dim=1), F.softmax(teacher_logit, dim=1))
                if self.ensemble_beta > 0:
                    loss = self.ensemble_alpha * teacher_loss - self.ensemble_beta * student_loss + self.ensemble_eta * diversity_loss
                else:
                    loss = self.ensemble_alpha * teacher_loss + self.ensemble_eta * diversity_loss
                # 反向传播
                loss.backward()
                self.generative_optimizer.step()
                TEACHER_LOSS += self.ensemble_alpha * teacher_loss  # (torch.mean(TEACHER_LOSS.double())).item()
                STUDENT_LOSS += self.ensemble_beta * student_loss  # (torch.mean(student_loss.double())).item()
                DIVERSITY_LOSS += self.ensemble_eta * diversity_loss  # (torch.mean(diversity_loss.double())).item()
            return TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS

        for i in range(epoches):
            TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS = update_generator_(
                self.n_teacher_iters, self.model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)

        TEACHER_LOSS = TEACHER_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        STUDENT_LOSS = STUDENT_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        DIVERSITY_LOSS = DIVERSITY_LOSS.detach().numpy() / (self.n_teacher_iters * epoches)
        info = "Generator: Teacher Loss= {:.4f}, Student Loss= {:.4f}, Diversity Loss = {:.4f}, ". \
            format(TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
        if verbose:
            print(info)
        self.generative_lr_scheduler.step()

    def get_label_weights(self):
        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):
            weights = []
            for user in self.selected_users:
                weights.append(user.label_counts[label])
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:
                qualified_labels.append(label)
            # uniform
            label_weights.append(np.array(weights) / np.sum(weights))
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        return label_weights, qualified_labels

    def visualize_images(self, generator, glob_iter, repeats=1):
        """
        Generate and visualize data for a generator.
        """
        os.system("mkdir -p images")
        path = f'images/{self.algorithm}-{self.dataset}-iter{glob_iter}.png'
        y = self.available_labels
        y = np.repeat(y, repeats=repeats, axis=0)
        y_input = torch.tensor(y)
        generator.eval()
        images = generator(y_input, latent=False)['output']  # 0,1,..,K, 0,1,...,K
        images = images.view(repeats, -1, *images.shape[1:])
        images = images.view(-1, *images.shape[2:])
        save_image(images.detach(), path, nrow=repeats, normalize=True)
        print("Image saved to {}".format(path))
