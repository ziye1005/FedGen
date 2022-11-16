#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
from FLAlgorithms.users.userbase import UserBase


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    # gt_mask是全0数组上为1处是标签，other_mask全1数组上为0处为标签
    other_mask = _get_other_mask(logits_student, target)
    pred_student0 = F.softmax(logits_student / temperature, dim=1)
    # pt(S)
    pred_teacher0 = F.softmax(logits_teacher / temperature, dim=1)
    # pt(T)
    pred_student = cat_mask(pred_student0, gt_mask, other_mask)
    # pred_student（pt_stu,pnt_stu）与pred_teacher(pt_tea,pnt_tea)这两列都是一个串联
    pred_teacher = cat_mask(pred_teacher0, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    a = F.kl_div(log_pred_student, pred_teacher, size_average=False)
    b = temperature**2
    c = target.shape[0]
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    # 这里面一个kl_div其实就是等式（4）的前两部分的和，即kl(bT || bS)，放在第一位的log_pred_student即bS是作为被指导的分子，放在第二位的pred_teacher即bT是作为出现两次的参数
    # **表示乘方，此处除以两个temperature
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 10.0 * gt_mask, dim=1
    )
    aa = logits_teacher / temperature
    bb = 1000.0 * gt_mask
    cc = aa - bb
    dd = F.softmax(aa, dim=1)
    ee = F.softmax(cc, dim=1)
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 10.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    # 变成一行
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    # one_hot操作，128*100的数组，标签处设为1，其余为0，128为batchsize抓取的数据，100为标签总数

    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    # reshape(-1)变为一列，128*1
    index = target.unsqueeze(1)
    # unsqueeze增加一列变成128*1*1，index有128行，每行都是一个1*1数组。
    # ones_like纯1数组，接下来类似one_hot操作，标签处设为0，其余位置为1
    mask1 = torch.ones_like(logits).scatter_(1, index, 0)
    # logit 128*100，torch.ones_like(logits)形成一个128*100的全1数组，假设128个数据的标签为[54,32,.....]，scatter_()函数将one_like()[0][54]变为0，[1][32]变为0
    mask =mask1.bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    # t1是p(t)的teacher或者student
    t2 = (t * mask2).sum(1, keepdims=True)
    # t2是p\(t)的teacher或者student
    rt = torch.cat([t1, t2], dim=1)
    return rt


class UserFedDKDGen(UserBase):
    # 在serverFedGen.init里面生成了传过来的generative_model，latent_layer_idx，available_labels，label_info，
    # 并在init 75行创建了total_users 20个用户，所以传给所有用户的以上四个参数都是相同的，但是用户自身的train_data和test_data不同
    def __init__(self,
                 args, id, model, generative_model,
                 train_data, test_data,
                 available_labels, latent_layer_idx, label_info,
                 use_adam=False, y_train=None):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)
        if y_train is None:
            y_train = []
        self.gen_batch_size = args.gen_batch_size
        self.generative_model = generative_model
        self.latent_layer_idx = latent_layer_idx
        self.available_labels = available_labels
        self.label_info = label_info
        self.temperature = args.temperature
        self.y_train = y_train
        self.distillAlpha = args.distillAlpha
        self.distillBeta = args.distillBeta
        # print('新增target ground truth标签', y_train)

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        # 每lr_decay_epoch epochs学习率衰减系数为0.95
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr = max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label: 1 for label in range(self.unique_labels)}

    # 对选择的每个user进行的FedGen.的train
    # 循环local_epochs，
    def train(self, glob_iter, personalized=False, early_stop=100, regularization=True, verbose=False):
        self.clean_up_counts()
        self.model.train()
        # eval 返回传入字符串的表达式的结果。就是说：将字符串当成有效的表达式 来求值并返回计算结果。
        self.generative_model.eval()
        TEACHER_LOSS, DIST_LOSS, LATENT_LOSS = 0, 0, 0
        # DIST_LOSS没用到
        for epoch in range(self.local_epochs):
            self.model.train()
            # 通过Net.py继承,调用的是torch自带，nn.module.train()
            for i in range(self.K):     # 默认是1
                # sample from real dataset (un-weighted)
                result = self.get_next_train_batch(count_labels=True)
                X, y = result['X'], result['y']
                self.update_label_counts(result['labels'], result['counts'])    # 更新本地标签分布
                self.optimizer.zero_grad()
                # 与FedAvg类似
                model_result = self.model(X, logit=True, temperature=self.temperature)
                # 改变本地模型，调用Net.forward()，返回的是results['output']表示log_softmax损失函数 + results['logit']表示predict layer，
                # Net.output_dim是从model_config.py里面获得的，log_softmax输出的结果，作为student
                user_output_logp = model_result['output']
                logits_student = model_result['logit']
                predictive_loss = self.loss(user_output_logp, y)    # 计算predict层的损失
                # 不同开始
                # sample y and generate z
                # 第一轮glob_iter所有user的regularization是false；后面轮次全是true；第一轮的所有user的loss相当于FedAvg的
                if regularization and epoch < early_stop:
                    # 学习率alpha和beta的优化更新，使用0.98的衰减系数 exp_lr_scheduler
                    generative_alpha = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_alpha)
                    generative_beta = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_beta)
                    generative_alpha_dkd = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.distillAlpha)
                    generative_beta_dkd = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.distillBeta)
                    # 获取相同标签的生成器输出(潜在表示)，generative_model是在serverFedGen里面初始化的Generator类的对象
                    gen_output = self.generative_model(y, latent_layer_idx=self.latent_layer_idx)['output']
                    # 调用Generator.forward()，返回输出层的信息
                    logit_given_gen = self.model(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)['logit']
                    # 改变本地模型，调用Net.mapping()，返回results['output']表示log_softmax损失函数 + results['logit']表示predict layer，
                    #
                    # target_p = F.softmax(logit_given_gen / self.temperature, dim=1).clone().detach()
                    # 这个user_latent_loss应该是kd_loss,user_output_logp是student，target_p是teacher
                    # user_latent_loss = generative_beta * self.ensemble_loss(user_output_logp, target_p)
                    loss_dkd = dkd_loss(logits_student, logit_given_gen, y, generative_alpha_dkd, generative_beta_dkd, self.temperature)
                    user_loss2 = generative_beta*loss_dkd
                    # temp = 10
                    # distillation_loss = self.soft_loss(F.softmax(student_preds / self.temperature, dim=1), F.softmax(teacher_preds/self.temperature, dim=1))
                    # 计算client的损失

                    sampled_y = np.random.choice(self.available_labels, self.gen_batch_size)
                    sampled_y = torch.tensor(sampled_y)
                    gen_result = self.generative_model(sampled_y, latent_layer_idx=self.latent_layer_idx)   # 调用Generator.forward()

                    gen_output = gen_result['output']  # latent representation when latent = True, x otherwise
                    user_output_logp = self.model(gen_output, start_layer_idx=self.latent_layer_idx)['output']
                    # 改变本地模型，依然调用Net.mapping()，teacher的loss
                    teacher_loss = generative_alpha * torch.mean(
                        self.generative_model.crossentropy_loss(user_output_logp, sampled_y)
                    )
                    # this is to further balance oversampled down-sampled synthetic data
                    gen_ratio = self.gen_batch_size / self.batch_size
                    loss = predictive_loss + gen_ratio * teacher_loss + user_loss2
                    TEACHER_LOSS += teacher_loss  # 用于打印
                    LATENT_LOSS += user_loss2
                else:
                    # get loss and perform optimization
                    loss = predictive_loss
                # 不同结束
                loss.backward()
                self.optimizer.step()  # self.local_model
        # optimizer.step()函数的作用是执行一次优化步骤，通过梯度下降法来更新参数的值。
        # local_model <=== self.model
        self.clone_model_paramenter(self.model.parameters(), self.local_model)
        if personalized:
            self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
        self.lr_scheduler.step(glob_iter)
        # 不同，输出
        if regularization and verbose:
            TEACHER_LOSS = TEACHER_LOSS.detach().numpy() / (self.local_epochs * self.K)
            LATENT_LOSS = LATENT_LOSS.detach().numpy() / (self.local_epochs * self.K)
            info = 'userFedGen verbose print\nUser Teacher Loss={:.4f}'.format(TEACHER_LOSS)
            info += ', Latent Loss={:.4f}'.format(LATENT_LOSS)
            info += ', generate_alpha={:.4f}'.format(generative_alpha)
            print(info)
        # 不同结束

    def adjust_weights(self, samples):
        labels, counts = samples['labels'], samples['counts']
        # weight=self.label_weights[y][:, user_idx].reshape(-1, 1)
        np_y = samples['y'].detach().numpy()
        n_labels = samples['y'].shape[0]
        weights = np.array([n_labels / count for count in counts])  # smaller count --> larger weight
        weights = len(self.available_labels) * weights / np.sum(weights)  # normalized
        label_weights = np.ones(self.unique_labels)
        label_weights[labels] = weights
        sample_weights = label_weights[np_y]
        return sample_weights

