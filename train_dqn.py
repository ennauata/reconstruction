import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from rl.environment import Environment
from options import parse_args
import os
from torch_geometric.nn import SplineConv, GCNConv
import cv2
import sys
from validator import Validator
from dataset.graph_dataloader import GraphData
from torch.utils.data import DataLoader
from tqdm import tqdm


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

GAMMA = 0.999
EPS_START = 0.9
#EPS_END = 0.05
EPS_END = 0.5
EPS_DECAY = 200
TARGET_UPDATE = 10

steps_done = 0

def train(options):
    model = GraphModel(options)
    model = model.cuda()
    model = model.train()

    # select optimizer
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=options.LR)

    PREFIX = options.data_path
    with open('{}/building_reconstruction/la_dataset_new/train_list_prime.txt'.format(PREFIX)) as f:
        file_list = [line.strip() for line in f.readlines()]
        train_list = file_list[:-50]
        valid_list = file_list[-50:]
        pass

    best_score = 0.0

    ##############################################################################################################
    ############################################### Start Training ###############################################
    ##############################################################################################################

    validator = Validator(options)

    num_edges = 0
    if options.restore == 1:
        model.load_state_dict(torch.load(options.checkpoint_dir + '/' + str(num_edges) + '_checkpoint.pth'))
        optimizer.load_state_dict(torch.load(options.checkpoint_dir + '/' + str(num_edges) + '_optim.pth'))
    elif options.restore == 2 and num_edges > 0:
        model.load_state_dict(torch.load(options.checkpoint_dir + '/' + str(num_edges - 1) + '_checkpoint.pth'))
        optimizer.load_state_dict(torch.load(options.checkpoint_dir + '/' + str(num_edges - 1) + '_optim.pth'))
    elif options.restore == 3:
        model.load_state_dict(torch.load(options.checkpoint_dir.replace('dets_only', 'annots_only') + '/' + str(num_edges) + '_checkpoint.pth'))
        optimizer.load_state_dict(torch.load(options.checkpoint_dir.replace('dets_only', 'annots_only') + '/' + str(num_edges) + '_optim.pth'))
        pass        

    #dset_val = GraphData(options, valid_list, split='val', num_edges=num_edges)
    dset_train = GraphData(options, train_list, num_edges=num_edges)
    #train_loader = DataLoader(dset_train, batch_size=64, shuffle=True, num_workers=1, collate_fn=PadCollate())

    for epoch in range(100):
        #os.system('rm ' + options.test_dir + '/' + str(num_edges) + '_*')
        dset_train.reset()
        train_loader = DataLoader(dset_train, batch_size=1, shuffle=True, num_workers=1)    
        epoch_losses = []
        ## Same with train_loader but provide progress bar
        data_iterator = tqdm(train_loader, total=len(dset_train))
        optimizer.zero_grad()        
        for sample_index, sample in enumerate(data_iterator):

            im_arr, corners, connections, corner_gt, connection_gt, edge_index, edge_attr, building_index = sample[0].cuda().squeeze(0), sample[1].cuda().squeeze(0), sample[2].cuda().squeeze(0), sample[3].cuda().squeeze(0), sample[4].cuda().squeeze(0), sample[5].cuda().squeeze(0), sample[6].cuda().squeeze(0), sample[7].squeeze().item()

            connection_confidence = validator.validate(dset_train.buildings[building_index])
            connections = torch.cat([connections, connection_confidence.unsqueeze(-1)], dim=-1)

            # corner_pred, connection_pred = model(corners, connections, edge_index, edge_attr)
            # corner_loss = F.binary_cross_entropy(corner_pred, corner_gt) * 0
            # connection_loss = F.binary_cross_entropy(connection_pred, connection_gt)
            # losses = [corner_loss, connection_loss]

            initial_state = (connection_confidence > 0.5).int()
            indices = torch.randint(len(initial_state), size=(5, ))
            state = initial_state.clone()
            state[indices] = 1 - state[indices]
            flipping_gt = (state.int() != (connection_gt > 0.5).int()).float()
            flipping_pred = model(state.float(), None, [connections, edge_index, edge_attr])
            flipping_loss = F.binary_cross_entropy(flipping_pred, flipping_gt)
            losses = [flipping_loss]
            connection_pred = state.clone()
            connection_pred[flipping_pred > 0.5] = 1 - connection_pred[flipping_pred > 0.5]
            loss = sum(losses)        

            loss.backward()

            if (sample_index + 1) % options.batch_size == 0:
                ## Progress bar
                loss_values = [l.data.item() for l in losses]
                epoch_losses.append(loss_values)
                status = str(epoch + 1) + ' loss: '
                for l in loss_values:
                    status += '%0.5f '%l
                    continue
                data_iterator.set_description(status)

                optimizer.step()
                optimizer.zero_grad()
                pass

            if sample_index % 1000 < 16:
                index_offset = sample_index % 1000
                building = dset_train.buildings[building_index]
                building.reset()
                building.update_edges(connection_pred.detach().cpu().numpy() > 0.5)
                images, _ = building.visualize(mode='last')
                cv2.imwrite(options.test_dir + '/' + str(index_offset) + '_image.png', images[0])
                building.reset()
                building.update_edges(state.detach().cpu().numpy() > 0.5)
                images, _ = building.visualize(mode='last')
                cv2.imwrite(options.test_dir + '/' + str(index_offset) + '_input.png', images[0])
                pass
            continue

        print('loss', np.array(epoch_losses).mean(0))
        # if epoch % 10 == 0:
        #     torch.save(model.state_dict(), options.checkpoint_dir + '/checkpoint_' + str(epoch // 10) + '.pth')
        #     torch.save(semantic_model.state_dict(), options.checkpoint_dir + '/checkpoint_semantic_' + str(epoch // 10) + '.pth')                
        #     #torch.save(optimizer.state_dict(), options.checkpoint_dir + '/optim_' + str(epoch // 10) + '.pth')
        #     pass
        torch.save(model.state_dict(), options.checkpoint_dir + '/' + str(num_edges) + '_checkpoint.pth')
        torch.save(optimizer.state_dict(), options.checkpoint_dir + '/' + str(num_edges) + '_optim.pth')
        #testOneEpoch(options, model, validator, dset_val)        
        continue
    return

def select_action(policy_net, state, env):
    global steps_done
    sample = np.random.random()
    
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = policy_net(state, env).max(-1)[1].view(1, 1)
            #print(action, sample, eps_threshold)
    else:
        #return torch.tensor([[np.random.randrange(len(state.connections))]], device=device, dtype=torch.long)
        action = torch.randint(len(state) + 1, size=(1, )).long().cuda().view(1, 1)
        pass
    #print('random', sample > eps_threshold, action)
    return action

class GraphModel(torch.nn.Module):
    def __init__(self, conv_type='spline'):
        super(GraphModel, self).__init__()
        #self.corner_encoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())
        #self.connection_encoder = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())
        #self.corner_encoder = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        #self.connection_encoder = nn.Sequential(nn.Linear(6, 32), nn.ReLU())        

        if conv_type == 'gcn':
            self.conv_1 = GCNConv(6, 32)
            self.conv_2 = GCNConv(32, 64)
            self.conv_3 = GCNConv(64, 64)
        else:
            self.conv_1 = SplineConv(6, 32, dim=1, kernel_size=2)
            self.conv_2 = SplineConv(32, 64, dim=1, kernel_size=2)
            self.conv_3 = SplineConv(64, 64, dim=1, kernel_size=2)
            pass

        #self.corner_pred = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.connection_pred = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        return
    
    def forward(self, states, env, sample=[]):
        #corner_x = self.corner_encoder(torch.cat([corners, torch.zeros(len(corners), 1).cuda()], dim=-1))
        #connection_x = self.connection_encoder(torch.cat([connections, torch.ones(len(connections), 1).cuda()], dim=-1))
        #connection_x = self.connection_encoder(connections)
        #x = torch.cat([corner_x, connection_x], 0)
        if len(sample) > 0:
            node_features, edge_index, edge_attr = sample
            node_features = torch.cat([node_features, states.unsqueeze(-1)], dim=-1)
            x = self.conv_1(node_features, edge_index, edge_attr)
            x = self.conv_2(x, edge_index, edge_attr)
            x = self.conv_3(x, edge_index, edge_attr)                
            preds = torch.sigmoid(self.connection_pred(x)).view(-1)
        else:
            if len(states.shape) == 1:
                states = states.unsqueeze(0)
                pass
            preds = []
            for state in states:
                node_features, edge_index, edge_attr = env.create_sample_graph(state)
                #print(state)
                x = self.conv_1(node_features, edge_index, edge_attr)
                x = self.conv_2(x, edge_index, edge_attr)
                x = self.conv_3(x, edge_index, edge_attr)                
                pred = torch.sigmoid(self.connection_pred(x)).view(-1)
                preds.append(pred)
                continue

            #return torch.sigmoid(self.corner_pred(x[:len(corners)])).view(-1), torch.sigmoid(self.connection_pred(x[len(corners):])).view(-1)
            preds = torch.stack(preds, dim=0)
            preds = torch.cat([preds, torch.ones((len(preds), 1)).cuda() * 0.5], dim=-1)
            # if probs.shape[0] == 1:
            #     probs = probs.squeeze(0)
            #     pass
            pass
        return preds


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return [self.memory[index] for index in np.random.choice(np.arange(len(self.memory), dtype=np.int32), batch_size)]

    def __len__(self):
        return len(self.memory)


def main(options):
    env = Environment(options)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env.reset()

    policy_net = GraphModel()
    policy_net.cuda()

    num_edges = 0
    if options.restore == 1:
        #policy_net.load_state_dict(torch.load(options.checkpoint_dir + '/' + str(num_edges) + '_checkpoint.pth'))
        policy_net.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint.pth'))
        pass
    
    target_net = GraphModel()
    target_net.cuda()    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)


    episode_durations = []


    def optimize_model():
        if len(memory) < options.batch_size:
            return
        transitions = memory.sample(options.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.uint8).cuda()
        non_final_next_states = torch.stack([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch, env).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(options.batch_size).cuda()
        next_state_values[non_final_mask] = target_net(non_final_next_states, env).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        
        # for c in range(16):
        #     if action_batch[c].item() < len(batch.next_state[c]):
        #         print(c, action_batch[c].item(), batch.next_state[c][action_batch[c]].item(), reward_batch[c].item(), state_batch[c])
        #         pass
        #     continue
        # print(state_action_values, expected_state_action_values)
        # exit(1)
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            continue
        optimizer.step()
        return

    ######################################################################
    #
    # Below, you can find the main training loop. At the beginning we reset
    # the environment and initialize the ``state`` Tensor. Then, we sample
    # an action, execute it, observe the next screen and the reward (always
    # 1), and optimize our model once. When the episode ends (our model
    # fails), we restart the loop.
    #
    # Below, `num_episodes` is set small. You should download
    # the notebook and run lot more epsiodes, such as 300+ for meaningful
    # duration improvements.
    #

    num_episodes = 300
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        state = env.get_initial_state()        
        cv2.imwrite(options.test_dir + '/image.png', env.visualize(state.detach().cpu().numpy()))
        
        for t in count():
            # Select and perform an action
            action = select_action(policy_net, state, env)
            next_state, reward, done, _ = env.step(state.detach(), action)
            reward = torch.tensor([reward]).cuda()

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
            sys.stdout.write('\r' + str(t) + ' ' + str(reward.item()) + ' ' + str(len(state) - action.item()))
            if done:
                episode_durations.append(t + 1)
                #plot_durations()
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            state = env.get_initial_state()
            for _ in range(10):
                action = policy_net(state, env).max(-1)[1].view(-1)
                next_state, reward, done, _ = env.step(state, action)
                print('inference', action.item(), next_state[min(action, len(next_state) - 1)].item(), reward)
                if action == len(state):
                    break
                state = next_state
                continue
            cv2.imwrite(options.test_dir + '/' + str(i_episode) + '_result.png', env.visualize(state.detach().cpu().numpy()))
            exit(1)            
            print('\nduration', min(episode_durations), max(episode_durations), float(sum(episode_durations)) / len(episode_durations))            
            pass
        continue
    return
        
if __name__ == '__main__':
    args = parse_args()
    args.keyname = 'dqn'
    args.keyname += '_' + args.corner_type
    if args.suffix != '':
        args.keyname += '_' + args.suffix
        pass
    if args.conv_type != '':
        args.keyname += '_' + args.conv_type
        pass    
    args.test_dir = 'test/' + args.keyname
    args.checkpoint_dir = 'checkpoint/' + args.keyname

    if not os.path.exists(args.checkpoint_dir):
        os.system("mkdir -p %s"%args.checkpoint_dir)
        pass
    if not os.path.exists(args.test_dir):
        os.system("mkdir -p %s"%args.test_dir)
        pass
    if not os.path.exists(args.test_dir + '/cache'):
        os.system("mkdir -p %s"%args.test_dir + '/cache')
        pass

    train(args)
    exit(1)
    main(args)
    
