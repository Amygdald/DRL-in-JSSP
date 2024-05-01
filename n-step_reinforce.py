import time
import random
import numpy as np
import numpy.random
from parameters import args
import torch
import torch.optim as optim
from env.environment import JsspN5, BatchGraph
from model.actor import Actor, Critic
from env.generateJSP import uni_instance_gen
from pathlib import Path
import copy



class RL2S4JSSP:
    def __init__(self):
        self.env_training = JsspN5(n_job=args.j, n_mch=args.m, low=args.l, high=args.h, reward_type=args.reward_type)
        self.env_validation = JsspN5(n_job=args.j, n_mch=args.m, low=args.l, high=args.h, reward_type=args.reward_type)
        self.eps = np.finfo(np.float32).eps.item()
        self.loss=[]
        validation_data_path = Path(
            './validation_data/validation_instance_{}x{}[{},{}].npy'.format(args.j, args.m, args.l, args.h))
        if validation_data_path.is_file():
            self.validation_data = np.load(
                './validation_data/validation_instance_{}x{}[{},{}].npy'.format(args.j, args.m, args.l, args.h))
        else:
            print('No validation data for {}x{}[{},{}], generating new one.'.format(args.j, args.m, args.l, args.h))
            self.validation_data = np.array(
                [uni_instance_gen(n_j=args.j, n_m=args.m, low=args.l, high=args.h) for _ in range(100)])
            np.save('./validation_data/validation_instance_{}x{}[{},{}].npy'.format(args.j, args.m, args.l, args.h),
                    self.validation_data)
        self.incumbent_validation_result = np.inf
        self.current_validation_result = np.inf

        if args.embedding_type == 'gin':
            self.dghan_param_for_saved_model = 'NAN'
        elif args.embedding_type == 'dghan' or args.embedding_type == 'gin+dghan':
            self.dghan_param_for_saved_model = '{}_{}'.format(args.heads, args.drop_out)
        else:
            raise Exception('embedding_type should be one of "gin", "dghan", or "gin+dghan".')

    def learn(self, rewards, log_probs, dones, optimizer):
        R = torch.zeros_like(rewards[0], dtype=torch.float, device=rewards[0].device)
        returns = []
        for r in rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)
        returns = torch.cat(returns, dim=-1)
        dones = torch.cat(dones, dim=-1)
        log_probs = torch.cat(log_probs, dim=-1)

        losses = []
        for b in range(returns.shape[0]):
            masked_R = torch.masked_select(returns[b], ~dones[b])
            masked_R = (masked_R - masked_R.mean()) / (torch.std(masked_R, unbiased=False) + self.eps)
            masked_log_prob = torch.masked_select(log_probs[b], ~dones[b])
            loss = (- masked_log_prob * masked_R).sum()
            losses.append(loss)


        optimizer.zero_grad()
        mean_loss = torch.stack(losses).mean()
        mean_loss.backward()
        # self.loss.append(mean_loss.item())
        optimizer.step()

    def validation(self, policy, dev):

        # fixed seed for fair validation: validation improve not because of different critical path
        # random.seed(1)
        # np.random.seed(1)
        # torch.manual_seed(1)

        # validating...
        validation_start = time.time()
        validation_batch_data = BatchGraph()
        states_val, feasible_actions_val, _ = self.env_validation.reset(instances=self.validation_data,
                                                                        init_type=args.init_type,
                                                                        device=dev)
        while self.env_validation.itr < args.transit:
            validation_batch_data.wrapper(*states_val)
            actions_val,_,_ = policy(validation_batch_data, feasible_actions_val)
            states_val, _, feasible_actions_val, _ = self.env_validation.step(actions_val, dev)
        states_val, feasible_actions_val, actions_val, _ = None, None, None, None
        validation_batch_data.clean()
        validation_result1 = self.env_validation.incumbent_objs.mean().cpu().item()
        validation_result2 = self.env_validation.current_objs.mean().cpu().item()
        # saving model based on validation results
        if validation_result1 < self.incumbent_validation_result:
            print('Find better model w.r.t incumbent objs, saving model...')
        #     torch.save(policy.state_dict(),
        #                './saved_model/incumbent_'  # saved model type
        #                '{}x{}[{},{}]_{}_{}_{}_'  # env parameters
        #                '{}_{}_{}_{}_{}_'  # model parameters
        #                '{}_{}_{}_{}_{}_{}'  # training parameters
        #                '.pth'
        #                .format(args.j, args.m, args.l, args.h, args.init_type, args.reward_type, args.gamma,
        #                        args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type, self.dghan_param_for_saved_model,
        #                        args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes, args.step_validation))
            self.incumbent_validation_result = validation_result1
        if validation_result2 < self.current_validation_result:
            print('Find better model w.r.t final step objs, saving model...')
            # torch.save(policy.state_dict(),
            #            './saved_model/last-step_'  # saved model type
            #            '{}x{}[{},{}]_{}_{}_{}_'  # env parameters
            #            '{}_{}_{}_{}_{}_'  # model parameters
            #            '{}_{}_{}_{}_{}_{}'  # training parameters
            #            '.pth'
            #            .format(args.j, args.m, args.l, args.h, args.init_type, args.reward_type, args.gamma,
            #                    args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type, self.dghan_param_for_saved_model,
            #                    args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes, args.step_validation))
            self.current_validation_result = validation_result2

        validation_end = time.time()

        print('Incumbent objs and final step objs for validation are: {:.2f}  {:.2f}'.format(validation_result1,
                                                                                             validation_result2),
              'validation takes:{:.2f}'.format(validation_end - validation_start))

        return validation_result1, validation_result2

    def train(self):
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'

        torch.manual_seed(1)
        random.seed(1)
        np.random.seed(1)

        policy = Actor(in_dim=3,
                       hidden_dim=args.hidden_dim,
                       embedding_l=args.embedding_layer,
                       policy_l=args.policy_layer,
                       embedding_type=args.embedding_type,
                       heads=args.heads,
                       dropout=args.drop_out).to(dev)

        optimizer = optim.Adam(policy.parameters(), lr=args.lr)

        batch_data = BatchGraph()
        log = []
        validation_log = []

        print()
        for batch_i in range(1, args.episodes // args.batch_size + 1):
        # for batch_i in range(1, 100):

            # random.seed(batch_i)
            # np.random.seed(batch_i)

            t1 = time.time()

            instances = np.array([uni_instance_gen(args.j, args.m, args.l, args.h) for _ in range(args.batch_size)])
            states, feasible_actions, dones = self.env_training.reset(instances=instances, init_type=args.init_type, device=dev)
            # print(instances)

            reward_log = []
            rewards_buffer = []
            log_probs_buffer = []
            dones_buffer = [dones]

            while self.env_training.itr < args.transit:
                batch_data.wrapper(*states)
                actions, log_ps,_ = policy(batch_data, feasible_actions)
                states, rewards, feasible_actions, dones = self.env_training.step(actions, dev)

                # store training data
                rewards_buffer.append(rewards)
                log_probs_buffer.append(log_ps)
                dones_buffer.append(dones)

                # logging reward...
                # reward_log.append(rewards)

                if self.env_training.itr % args.steps_learn == 0:
                # if self.env_training.itr % 10 == 0:
                    # training...
                    self.learn(rewards_buffer, log_probs_buffer, dones_buffer[:-1], optimizer)
                    # clean training data
                    rewards_buffer = []
                    log_probs_buffer = []
                    dones_buffer = [dones]

            # learn(rewards_buffer, log_probs_buffer, dones_buffer[:-1])  # old-school training scheme

            t2 = time.time()
            print('Batch {} training takes: {:.2f}'.format(batch_i, t2 - t1),
                  'Mean Performance: {:.2f}'.format(self.env_training.current_objs.cpu().mean().item()))
            log.append(self.env_training.current_objs.mean().cpu().item())

            # start validation and saving model & logs...
            if batch_i % args.step_validation == 0:
                # validating...
                validation_result1, validation_result2 = self.validation(policy, dev)
                validation_log.append([validation_result1, validation_result2])
                # p_loss_cpu = [np.array(tensor) for tensor in self.loss]
                # np.save('./log/p_loss', np.array(self.loss))

                # saving log
                np.save('./test_log/oldtraining_log_'
                        '{}x{}[{},{}]_{}_{}_{}_'  # env parameters
                        '{}_{}_{}_{}_{}_'  # model parameters
                        '{}_{}_{}_{}_{}_{}.npy'  # training parameters
                        .format(args.j, args.m, args.l, args.h, args.init_type, args.reward_type, args.gamma,
                                args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type,
                                self.dghan_param_for_saved_model,
                                args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes,
                                args.step_validation),
                        np.array(log))
                # np.save('./test_log/validation_log_'
                #         '{}x{}[{},{}]_{}_{}_{}_'  # env parameters
                #         '{}_{}_{}_{}_{}_'  # model parameters
                #         '{}_{}_{}_{}_{}_{}.npy'  # training parameters
                #         .format(args.j, args.m, args.l, args.h, args.init_type, args.reward_type, args.gamma,
                #                 args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type,
                #                 self.dghan_param_for_saved_model,
                #                 args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes,
                #                 args.step_validation),
                #         np.array(validation_log))

class PPO:
    def __init__(self):
        self.clip_ratio = 0.2
        self.clip_v=True
        self.p_loss=[]
        self.v_loss=[]
        self.entropy=[]
        self.env_training = JsspN5(n_job=args.j, n_mch=args.m, low=args.l, high=args.h, reward_type=args.reward_type)
        self.env_validation = JsspN5(n_job=args.j, n_mch=args.m, low=args.l, high=args.h, reward_type=args.reward_type)
        self.eps = np.finfo(np.float32).eps.item()
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy = Actor(in_dim=3,
                       hidden_dim=args.hidden_dim,
                       embedding_l=args.embedding_layer,
                       policy_l=args.policy_layer,
                       embedding_type=args.embedding_type,
                       heads=args.heads,
                       dropout=args.drop_out).to(self.dev)
        self.critic= Critic(in_dim=3,
                       hidden_dim=args.hidden_dim,
                       embedding_l=args.embedding_layer,
                       # policy_l=args.policy_layer,
                       embedding_type=args.embedding_type,
                       heads=args.heads,
                       dropout=args.drop_out).to(self.dev)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=args.lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=args.lr/5)

        validation_data_path = Path(
            './validation_data/validation_instance_{}x{}[{},{}].npy'.format(args.j, args.m, args.l, args.h))
        if validation_data_path.is_file():
            self.validation_data = np.load(
                './validation_data/validation_instance_{}x{}[{},{}].npy'.format(args.j, args.m, args.l, args.h))
        else:
            print('No validation data for {}x{}[{},{}], generating new one.'.format(args.j, args.m, args.l, args.h))
            self.validation_data = np.array(
                [uni_instance_gen(n_j=args.j, n_m=args.m, low=args.l, high=args.h) for _ in range(100)])
            np.save('./validation_data/validation_instance_{}x{}[{},{}].npy'.format(args.j, args.m, args.l, args.h),
                    self.validation_data)
        self.incumbent_validation_result = np.inf
        self.current_validation_result = np.inf

        if args.embedding_type == 'gin':
            self.dghan_param_for_saved_model = 'NAN'
        elif args.embedding_type == 'dghan' or args.embedding_type == 'gin+dghan':
            self.dghan_param_for_saved_model = '{}_{}'.format(args.heads, args.drop_out)
        else:
            raise Exception('embedding_type should be one of "gin", "dghan", or "gin+dghan".')

    def validation(self, policy, dev):

        # fixed seed for fair validation: validation improve not because of different critical path
        # random.seed(1)
        # np.random.seed(1)
        # torch.manual_seed(1)

        # validating...
        validation_start = time.time()
        validation_batch_data = BatchGraph()
        states_val, feasible_actions_val, _ = self.env_validation.reset(instances=self.validation_data,
                                                                        init_type=args.init_type,
                                                                        device=dev)
        while self.env_validation.itr < args.transit:
            validation_batch_data.wrapper(*states_val)
            actions_val, _,_ = policy(validation_batch_data, feasible_actions_val)
            states_val, _, feasible_actions_val, _ = self.env_validation.step(actions_val, dev)
        states_val, feasible_actions_val, actions_val, _ = None, None, None, None
        validation_batch_data.clean()
        validation_result1 = self.env_validation.incumbent_objs.mean().cpu().item()
        validation_result2 = self.env_validation.current_objs.mean().cpu().item()
        # saving model based on validation results
        if validation_result1 < self.incumbent_validation_result:
            print('Find better model w.r.t incumbent objs, saving model...')
            torch.save(policy.state_dict(),
                       './new_saved_model/incumbent_'  # saved model type
                       '{}x{}[{},{}]_{}_{}_{}_'  # env parameters
                       '{}_{}_{}_{}_{}_'  # model parameters
                       '{}_{}_{}_{}_{}_{}'  # training parameters
                       '.pth'
                       .format(args.j, args.m, args.l, args.h, args.init_type, args.reward_type, args.gamma,
                               args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type, self.dghan_param_for_saved_model,
                               args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes, args.step_validation))
            self.incumbent_validation_result = validation_result1
        if validation_result2 < self.current_validation_result:
            print('Find better model w.r.t final step objs, saving model...')
            torch.save(policy.state_dict(),
                       './new_saved_model/last-step_'  # saved model type
                       '{}x{}[{},{}]_{}_{}_{}_'  # env parameters
                       '{}_{}_{}_{}_{}_'  # model parametersdsa
                       '{}_{}_{}_{}_{}_{}'  # training parameters
                       '.pth'
                       .format(args.j, args.m, args.l, args.h, args.init_type, args.reward_type, args.gamma,
                               args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type, self.dghan_param_for_saved_model,
                               args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes, args.step_validation))
            self.current_validation_result = validation_result2

        validation_end = time.time()

        print('Incumbent objs and final step objs for validation are: {:.2f}  {:.2f}'.format(validation_result1,
                                                                                             validation_result2),
              'validation takes:{:.2f}'.format(validation_end - validation_start))

        return validation_result1, validation_result2

    def new_train(self):

        torch.manual_seed(1)
        random.seed(1)
        np.random.seed(1)

        batch_data = BatchGraph()
        # self.oldpolicy.load_state_dict(self.policy.state_dict())
        log = []
        validation_log = []

        print()
        for batch_i in range(1, args.episodes // (args.batch_size) + 1):

            t1 = time.time()

            instances = np.array([uni_instance_gen(args.j, args.m, args.l, args.h) for _ in range(args.batch_size)])
            states, feasible_actions, dones = self.env_training.reset(instances=instances, init_type=args.init_type, device=self.dev)
            # print(instances)

            # reward_log = []
            rewards_buffer = []
            log_probs_buffer = []
            dones_buffer = [dones]
            states_buffer = []
            actions_buffer = []
            values_buffer = []

            while self.env_training.itr < (args.transit):
                # self.oldpolicy.eval()
                with torch.no_grad():
                    batch_data.wrapper(*states)
                    actions, log_ps,_ = self.policy(batch_data, feasible_actions)

                    values=self.critic(batch_data, feasible_actions)
                    states_buffer.append((states,feasible_actions))
                    actions_buffer.append(actions)

                    states, rewards, feasible_actions, dones = self.env_training.step(actions, self.dev)

                # store training data
                rewards_buffer.append(rewards)
                log_probs_buffer.append(log_ps)
                dones_buffer.append(dones)
                values_buffer.append(values)

                # self.reward.append(rewards)

                if self.env_training.itr % args.steps_learn == 0:
                    # training...
                    # random_indices = np.random.choice(args.steps_learn, args.update_batch, replace=False)
                    # states_buffer = [states_buffer[i] for i in random_indices]
                    # actions_buffer = [actions_buffer[i] for i in random_indices]
                    # rewards_buffer = [rewards_buffer[i] for i in random_indices]
                    # dones_buffer = [dones_buffer[:-1][i] for i in random_indices]
                    # values_buffer = [values_buffer[i] for i in random_indices]
                    # log_probs_buffer = [log_probs_buffer[i] for i in random_indices]
                    self.new_learn(batch_data,
                                   states_buffer,
                                   actions_buffer,
                                   log_probs_buffer,
                                   rewards_buffer,
                                   dones_buffer[:-1],
                                   values_buffer)
                    # clean training data
                    rewards_buffer = []
                    log_probs_buffer = []
                    dones_buffer = [dones]
                    states_buffer = []
                    actions_buffer = []
                    values_buffer = []

                    # self.oldpolicy.load_state_dict(self.policy.state_dict())

            t2 = time.time()
            print('Batch {} training takes: {:.2f}'.format(batch_i, t2 - t1),
                  'Mean Performance: {:.2f}'.format(self.env_training.current_objs.cpu().mean().item()))
            log.append(self.env_training.current_objs.mean().cpu().item())

            # start validation and saving model & logs...
            if batch_i % args.step_validation == 0:
                # validating...
                validation_result1, validation_result2 = self.validation(self.policy, self.dev)
                validation_log.append([validation_result1, validation_result2])

                np.save('./log_new/training_log_'
                        '{}x{}[{},{}]_{}_{}_{}_'  # env parameters
                        '{}_{}_{}_{}_{}_'  # model parameters
                        '{}_{}_{}_{}_{}_{}.npy'  # training parameters
                        .format(args.j, args.m, args.l, args.h, args.init_type, args.reward_type, args.gamma,
                                args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type,
                                self.dghan_param_for_saved_model,
                                args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes,
                                args.step_validation),
                        np.array(log))
                np.save('./log_new/validation_log_'
                        '{}x{}[{},{}]_{}_{}_{}_'  # env parameters
                        '{}_{}_{}_{}_{}_'  # model parameters
                        '{}_{}_{}_{}_{}_{}.npy'  # training parameters
                        .format(args.j, args.m, args.l, args.h, args.init_type, args.reward_type, args.gamma,
                                args.hidden_dim, args.embedding_layer, args.policy_layer, args.embedding_type,
                                self.dghan_param_for_saved_model,
                                args.lr, args.steps_learn, args.transit, args.batch_size, args.episodes,
                                args.step_validation),
                        np.array(validation_log))
                # p_loss_cpu = [tensor.cpu().detach().numpy() for tensor in self.p_loss]
                # v_loss_cpu = [tensor.cpu().detach().numpy() for tensor in self.v_loss]
                # en_cpu=[tensor.cpu().detach().numpy() for tensor in self.entropy]
                # np.save('./log_new/p_loss',np.array(p_loss_cpu))
                # np.save('./log_new/v_loss', np.array(v_loss_cpu))
                # np.save('./log_new/entropy',np.array(en_cpu))

                np.save('./log_new/p_loss', np.array(self.entropy))
                self.entropy=[]

    def new_learn(self,batch_data,states,actions,old_log_probs_buffer,rewards, dones,old_values):

        R = torch.zeros_like(rewards[0], dtype=torch.float, device=rewards[0].device)
        gae = torch.zeros_like(rewards[0], dtype=torch.float, device=rewards[0].device)
        returns = []
        adv=[]
        # for i,(r,v) in enumerate(zip(rewards[::-1],old_values[::-1])):
        for r in rewards[::-1]:
        #     if i==0: v_next=v
            R = r + args.gamma * R
            gae=R
            # delta = r + args.gamma * v_next - v
            # gae = delta + args.gamma * args.lmbda * gae
            #
            # R=gae+v
            #
            # v_next=v

            returns.insert(0, R)
            adv.insert(0,gae)

        # if self.clip_v:
        #     old_values=torch.cat(old_values, dim=-1)
        returns = torch.cat(returns, dim=-1)
        dones = torch.cat(dones, dim=-1)
        adv=torch.cat(adv, dim=-1)

        for _ in range(args.update_times):
            log_probs_buffer=[]
            new_values_buffer=[]
            dist_entropy_buffer=[]
            # old_log_probs_buffer=[]
            for (old_states,old_feasible_actions),old_actions in zip(states,actions):

                batch_data.wrapper(*old_states)
                critic_values = self.critic(batch_data, old_feasible_actions)
                log_ps, dist_entropy = self.policy.evaluate(batch_data, old_feasible_actions, old_actions)
                # old_log_ps,_=self.oldpolicy.evaluate(batch_data, old_feasible_actions,old_actions)
                # _,log_ps,_=self.policy(batch_data, old_feasible_actions)

                log_probs_buffer.append(log_ps)
                new_values_buffer.append(critic_values)
                dist_entropy_buffer.append(dist_entropy)
                # old_log_probs_buffer.append(old_log_ps.detach())

            log_probs = torch.cat(log_probs_buffer, dim=-1)
            old_log_probs_tensor = torch.cat(old_log_probs_buffer, dim=-1)
            # print(old_log_probs_tensor.requires_grad)
            # new_values = torch.cat(new_values_buffer, dim=-1)
            dist_entropy_buffer_tensor=torch.cat(dist_entropy_buffer, dim=-1)

            losses = []
            entropy_losses = []
            # value_losses=[]
            entropy=[]
            for b in range(returns.shape[0]):
                # if self.clip_v:
                #     masked_old=torch.masked_select(old_values[b], ~dones[b])
                # masked_R = torch.masked_select(returns[b], ~dones[b])
                # masked_value=torch.masked_select(new_values[b], ~dones[b])
                masked_log_prob = torch.masked_select(log_probs[b], ~dones[b])
                masked_old_log_prob = torch.masked_select(old_log_probs_tensor[b],~dones[b])
                masked_adv=torch.masked_select(adv[b], ~dones[b])
                masked_entropy=torch.masked_select(dist_entropy_buffer_tensor[b],~dones[b])

                # if (masked_R.nelement() !=args.steps_learn or masked_value.nelement() !=args.steps_learn or
                #         masked_log_prob.nelement() !=args.steps_learn or masked_old_log_prob.nelement() !=args.steps_learn or
                #         masked_adv.nelement() !=args.steps_learn):
                #     continue

                # masked_R = (masked_R - masked_R.mean()) / (torch.std(masked_R, unbiased=False) + self.eps)
                # masked_value = (masked_value - masked_value.mean()) / (
                #             torch.std(masked_value, unbiased=False) + self.eps)
                # masked_old=(masked_old - masked_old.mean()) / (
                #             torch.std(masked_old, unbiased=False) + self.eps)
                # if self.clip_v:
                #     clip_v = masked_old + torch.clamp(masked_value - masked_old, -self.clip_ratio, self.clip_ratio)
                #     v_max = torch.max(((masked_old - masked_R) ** 2), ((clip_v - masked_R) ** 2)).sum()

                masked_advantage = (masked_adv - masked_adv.mean()) / (torch.std(masked_adv, unbiased=False) + self.eps)

                ratio=torch.exp(masked_log_prob-masked_old_log_prob)
                # ratio=masked_log_prob
                surr1 = (ratio * masked_advantage)
                # surr2=surr1
                surr2 = (torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * masked_advantage)

                loss = (- torch.min(surr1, surr2)).sum()
                loss_entropy = masked_entropy.mean()

                losses.append(loss)
                entropy_losses.append(loss_entropy)
                # value_losses.append(v_max)
            # entropy.append(loss_entropy)

            self.optimizer_policy.zero_grad()
            mean_loss = torch.stack(losses).mean()
            mean_entropy = torch.stack(entropy_losses).mean()
            mean_loss=mean_loss+mean_entropy*args.beta
            mean_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            # self.p_loss.append(mean_loss)
            self.entropy.append(mean_entropy.item())
            self.optimizer_policy.step()

            # self.optimizer_critic.zero_grad()
            # mean_value_loss = torch.stack(value_losses).mean() * args.alpha
            # mean_value_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(),0.5)
            # # self.v_loss.append(mean_value_loss)
            # self.optimizer_critic.step()


if __name__ == '__main__':
    # agent = RL2S4JSSP()
    # agent.train()
    agent = PPO()
    agent.new_train()
