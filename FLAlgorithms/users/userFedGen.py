import torch
import torch.nn.functional as F
import numpy as np
from FLAlgorithms.users.userbase import UserBase


class UserFedGen(UserBase):
    # 在serverFedGen.init里面生成了传过来的generative_model，latent_layer_idx，available_labels，label_info，
    # 并在init 75行创建了total_users 20个用户，所以传给所有用户的以上四个参数都是相同的，但是用户自身的train_data和test_data不同
    def __init__(self,
                 args, id, model, generative_model,
                 train_data, test_data,
                 available_labels, latent_layer_idx, label_info,
                 use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)
        self.gen_batch_size = args.gen_batch_size
        self.generative_model = generative_model
        self.latent_layer_idx = latent_layer_idx
        self.available_labels = available_labels
        self.label_info = label_info

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
        self.generative_model.eval()    # 不同
        TEACHER_LOSS, DIST_LOSS, LATENT_LOSS = 0, 0, 0      # 不同
        for epoch in range(self.local_epochs):
            self.model.train()
            for i in range(self.K):
                # sample from real dataset (un-weighted)
                result = self.get_next_train_batch(count_labels=True)
                X, y = result['X'], result['y']
                self.update_label_counts(result['labels'], result['counts'])
                self.optimizer.zero_grad()
                # 不同
                model_result = self.model(X, logit=True)
                user_output_logp = model_result['output']   # 得到predict层的输出
                predictive_loss = self.loss(user_output_logp, y)    # 计算predict层的损失

                # sample y and generate z
                if regularization and epoch < early_stop:
                    # 学习率alpha和beta的优化更新，使用0.95的衰减系数 exp_lr_scheduler
                    generative_alpha = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_alpha)
                    generative_beta = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_beta)
                    # 获取相同标签的生成器输出(潜在表示)
                    gen_output = self.generative_model(y, latent_layer_idx=self.latent_layer_idx)['output']
                    logit_given_gen = self.model(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)['logit']
                    # 这里面应该是修改的地方
                    target_p = F.softmax(logit_given_gen, dim=1).clone().detach()
                    user_latent_loss = generative_beta * self.ensemble_loss(user_output_logp, target_p)

                    sampled_y = np.random.choice(self.available_labels, self.gen_batch_size)
                    sampled_y = torch.tensor(sampled_y)
                    gen_result = self.generative_model(sampled_y, latent_layer_idx=self.latent_layer_idx)
                    gen_output = gen_result['output']  # latent representation when latent = True, x otherwise
                    user_output_logp = self.model(gen_output, start_layer_idx=self.latent_layer_idx)['output']
                    teacher_loss = generative_alpha * torch.mean(
                        self.generative_model.crossentropy_loss(user_output_logp, sampled_y)
                    )
                    # this is to further balance oversampled down-sampled synthetic data
                    gen_ratio = self.gen_batch_size / self.batch_size
                    loss = predictive_loss + gen_ratio * teacher_loss + user_latent_loss
                    TEACHER_LOSS += teacher_loss
                    LATENT_LOSS += user_latent_loss
                else:
                    # get loss and perform optimization
                    loss = predictive_loss
                # 不同结束
                loss.backward()
                self.optimizer.step()  # self.local_model)
        # local-model <=== self.model
        self.clone_model_paramenter(self.model.parameters(), self.local_model)
        if personalized:
            self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
        self.lr_scheduler.step(glob_iter)
        # 不同
        if regularization and verbose:
            TEACHER_LOSS = TEACHER_LOSS.detach().numpy() / (self.local_epochs * self.K)
            LATENT_LOSS = LATENT_LOSS.detach().numpy() / (self.local_epochs * self.K)
            info = '\nUser Teacher Loss={:.4f}'.format(TEACHER_LOSS)
            info += ', Latent Loss={:.4f}'.format(LATENT_LOSS)
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
