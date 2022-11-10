import torch
import torch.nn.functional as F
import numpy as np
from FLAlgorithms.users.userbase import UserBase


class UserFedGen(UserBase):
    # ��serverFedGen.init���������˴�������generative_model��latent_layer_idx��available_labels��label_info��
    # ����init 75�д�����total_users 20���û������Դ��������û��������ĸ�����������ͬ�ģ������û������train_data��test_data��ͬ
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
        self.temperature = args.temperature

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        # ÿlr_decay_epoch epochsѧϰ��˥��ϵ��Ϊ0.95
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr = max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label: 1 for label in range(self.unique_labels)}

    # ��ѡ���ÿ��user���е�FedGen.��train
    # ѭ��local_epochs��
    def train(self, glob_iter, personalized=False, early_stop=100, regularization=True, verbose=False):
        self.clean_up_counts()
        self.model.train()
        # eval ���ش����ַ����ı��ʽ�Ľ��������˵�����ַ���������Ч�ı��ʽ ����ֵ�����ؼ�������
        self.generative_model.eval()
        TEACHER_LOSS, DIST_LOSS, LATENT_LOSS = 0, 0, 0
        # DIST_LOSSû�õ�
        for epoch in range(self.local_epochs):
            self.model.train()
            # ͨ��Net.py�̳�,���õ���torch�Դ���nn.module.train()
            for i in range(self.K):     # Ĭ����1
                # sample from real dataset (un-weighted)
                result = self.get_next_train_batch(count_labels=True)
                X, y = result['X'], result['y']
                self.update_label_counts(result['labels'], result['counts'])    # ���±��ر�ǩ�ֲ�
                self.optimizer.zero_grad()
                # ��FedAvg����
                model_result = self.model(X, logit=True, temperature=self.temperature)
                # �ı䱾��ģ�ͣ�����Net.forward()�����ص���results['output']��ʾlog_softmax��ʧ���� + results['logit']��ʾpredict layer��
                # Net.output_dim�Ǵ�model_config.py�����õ�
                user_output_logp = model_result['output']
                predictive_loss = self.loss(user_output_logp, y)    # ����predict�����ʧ
                # ��ͬ��ʼ
                # sample y and generate z
                # ��һ��glob_iter����user��regularization��false�������ִ�ȫ��true����һ�ֵ�����user��loss�൱��FedAvg��
                if regularization and epoch < early_stop:
                    # ѧϰ��alpha��beta���Ż����£�ʹ��0.98��˥��ϵ�� exp_lr_scheduler
                    generative_alpha = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_alpha)
                    generative_beta = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_beta)
                    # ��ȡ��ͬ��ǩ�����������(Ǳ�ڱ�ʾ)��generative_model����serverFedGen�����ʼ����Generator��Ķ���
                    gen_output = self.generative_model(y, latent_layer_idx=self.latent_layer_idx)['output']
                    # ����Generator.forward()��������������Ϣ
                    logit_given_gen = self.model(gen_output, start_layer_idx=self.latent_layer_idx, logit=True)['logit']
                    # �ı䱾��ģ�ͣ�����Net.mapping()������results['output']��ʾlog_softmax��ʧ���� + results['logit']��ʾpredict layer��
                    # ������Ӧ�����޸ĵĵط���KDѧ��������ĵ�һ�仰
                    target_p = F.log_softmax(logit_given_gen / self.temperature, dim=1).clone().detach()
                    user_latent_loss = generative_beta * self.ensemble_loss(user_output_logp, target_p)
                    # temp = 10
                    # distillation_loss = self.soft_loss(F.softmax(student_preds / self.temperature, dim=1), F.softmax(teacher_preds/self.temperature, dim=1))
                    # ����client����ʧ

                    sampled_y = np.random.choice(self.available_labels, self.gen_batch_size)
                    sampled_y = torch.tensor(sampled_y)
                    gen_result = self.generative_model(sampled_y, latent_layer_idx=self.latent_layer_idx)   # ����Generator.forward()

                    gen_output = gen_result['output']  # latent representation when latent = True, x otherwise
                    user_output_logp = self.model(gen_output, start_layer_idx=self.latent_layer_idx)['output']
                    # �ı䱾��ģ�ͣ���Ȼ����Net.mapping()��teacher��loss
                    teacher_loss = generative_alpha * torch.mean(
                        self.generative_model.crossentropy_loss(user_output_logp, sampled_y)
                    )
                    # this is to further balance oversampled down-sampled synthetic data
                    gen_ratio = self.gen_batch_size / self.batch_size
                    loss = predictive_loss + gen_ratio * teacher_loss + user_latent_loss
                    TEACHER_LOSS += teacher_loss  # ���ڴ�ӡ
                    LATENT_LOSS += user_latent_loss
                else:
                    # get loss and perform optimization
                    loss = predictive_loss
                # ��ͬ����
                loss.backward()
                self.optimizer.step()  # self.local_model
        # optimizer.step()������������ִ��һ���Ż����裬ͨ���ݶ��½��������²�����ֵ��
        # local_model <=== self.model
        self.clone_model_paramenter(self.model.parameters(), self.local_model)
        if personalized:
            self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
        self.lr_scheduler.step(glob_iter)
        # ��ͬ�����
        if regularization and verbose:
            TEACHER_LOSS = TEACHER_LOSS.detach().numpy() / (self.local_epochs * self.K)
            LATENT_LOSS = LATENT_LOSS.detach().numpy() / (self.local_epochs * self.K)
            info = 'userFedGen verbose print\nUser Teacher Loss={:.4f}'.format(TEACHER_LOSS)
            info += ', Latent Loss={:.4f}'.format(LATENT_LOSS)
            info += ', generate_alpha={:.4f}'.format(generative_alpha)
            print(info)
        # ��ͬ����

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
