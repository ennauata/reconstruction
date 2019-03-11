import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import glob
import numpy as np
import pickle as p
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset.graph_dataloader import GraphData
from dataset.shuffle_dataloader import ShuffleData
from torch.utils.data import DataLoader
from dataset.collate import PadCollate
from utils.losses import balanced_binary_cross_entropy
import os
from dataset.metrics import Metrics
from collections import OrderedDict
import cv2
from tqdm import tqdm
from utils.utils import tileImages
from validator import Validator
from options import parse_args
from model.edge import LoopModel
from model.modules import findLoopsModule

def main(options):
    ##############################################################################################################
    ############################################### Define Model #################################################
    ##############################################################################################################

    # edge_pred, edge_corner, num_corners, corners = torch.load('test/debug.pth')
    # print(edge_pred)
    # all_loops = findLoopsModule(edge_pred, edge_corner, num_corners, max_num_loop_corners=12, corners=corners, disable_colinear=True, disable_intersection=True)
    # exit(1)
    
    model = LoopModel(options)
    model = model.cuda()
    model = model.train()

    # select optimizer
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=options.LR)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    # sample_eval = EdgeClassifier()
    # sample_eval = sample_eval.cpu()
    # sample_eval = sample_eval.eval()
    ##############################################################################################################
    ############################################# Setup Training #################################################
    ##############################################################################################################
    PREFIX = options.data_path
    with open('{}/train_list.txt'.format(PREFIX)) as f:
        file_list = [line.strip() for line in f.readlines()]
        train_list = file_list
        # train_list = file_list[:-50]
        # valid_list = file_list[-50:]
        pass

    with open('{}/valid_list.txt'.format(PREFIX)) as f:
        valid_list = [line.strip() for line in f.readlines()]
        pass

    best_score = 0.0
    mt = Metrics()

    ##############################################################################################################
    ############################################### Start Training ###############################################
    ##############################################################################################################

    all_num_edges = [0]

    ## Train incrementally
    for num_edges in all_num_edges:
        print('num edges', num_edges)
        if options.restore == 1:
            model.load_state_dict(torch.load(options.checkpoint_dir + '/' + str(num_edges) + '_checkpoint.pth'))
            optimizer.load_state_dict(torch.load(options.checkpoint_dir + '/' + str(num_edges) + '_optim.pth'))
        elif options.restore == 2 and num_edges > 0:
            model.load_state_dict(torch.load(options.checkpoint_dir + '/' + str(num_edges - 1) + '_checkpoint.pth'))
            optimizer.load_state_dict(torch.load(options.checkpoint_dir + '/' + str(num_edges - 1) + '_optim.pth'))
        elif options.restore == 3:
            model.load_state_dict(torch.load('checkpoint/batch_annots_only/' + str(num_edges) + '_checkpoint.pth'))
            optimizer.load_state_dict(torch.load('checkpoint/batch_annots_only/' + str(num_edges) + '_optim.pth'))
            pass        
        elif options.restore == 4:
            state_dict = torch.load(options.checkpoint_dir + '/' + str(num_edges) + '_checkpoint.pth')
            state = model.state_dict()
            new_state_dict = {k: v for k, v in state_dict.items() if k in state and state[k].shape == v.shape}
            state.update(new_state_dict)
            model.load_state_dict(state)
            #optimizer.load_state_dict(torch.load(options.checkpoint_dir + '/' + str(num_edges) + '_optim.pth'))            
        if options.num_edges == -1 and os.path.exists(options.checkpoint_dir + '/' + str(num_edges) + '_checkpoint.pth'):
            continue

        dset_val = GraphData(options, valid_list, split='val', num_edges=num_edges, load_heatmaps=True)

        if options.task == 'test':
            with torch.no_grad():
                testOneEpoch(options, model, dset_val)
            exit(1)
            pass
        
        if options.task == 'visualize':
            with torch.no_grad():
                additional_models = []
                # if options.suffix == '':
                #     annotation_model = create_model(options)
                #     annotation_model = annotation_model.cuda()
                #     annotation_model.load_state_dict(torch.load(options.checkpoint_dir.replace(options.corner_type, 'annots_only') + '/' + str(num_edges) + '_checkpoint.pth'))
                #     additional_models.append(annotation_model)
                #     pass
                testOneEpoch(options, model, dset_val, additional_models, visualize=True)
                pass
            exit(1)
            pass


        dset_train = GraphData(options, train_list, num_edges=num_edges, load_heatmaps=True)
            
        #train_loader = DataLoader(dset_train, batch_size=64, shuffle=True, num_workers=1, collate_fn=PadCollate())

        if options.task == 'search':
            for split, dataset in [('train', dset_train), ('val', dset_val)]:
                all_statistics = np.zeros(options.max_num_edges + 1)
                for building in dataset.buildings:
                    building.reset(num_edges)
                    statistics = validator.search(building, num_edges_target=num_edges + 1)
                    all_statistics += statistics
                    if statistics.sum() < 1:
                        print(building._id, statistics.sum(), building.num_edges, building.num_edges_gt)
                        pass
                    building.save()
                    continue
                print(split, all_statistics / len(dataset.buildings))
                continue
            exit(1)
            pass

        # for m in model.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.eval()

        for epoch in range(100):
            #os.system('rm ' + options.test_dir + '/' + str(num_edges) + '_*')
            dset_train.reset()
            train_loader = DataLoader(dset_train, batch_size=1, shuffle=True, num_workers=1)    
            epoch_losses = []
            ## Same with train_loader but provide progress bar
            data_iterator = tqdm(train_loader, total=len(dset_train))
            optimizer.zero_grad()        
            for sample_index, sample in enumerate(data_iterator):

                im_arr, corner_images, edge_images, corners, edges, corner_gt, edge_gt, corner_edge_pairs, edge_corner, left_edges, right_edges, building_index = sample[0].cuda().squeeze(0), sample[1].cuda().squeeze(0), sample[2].cuda().squeeze(0), sample[3].cuda().squeeze(0), sample[4].cuda().squeeze(0), sample[5].cuda().squeeze(0), sample[6].cuda().squeeze(0), sample[7].cuda().squeeze(), sample[8].cuda().squeeze(), sample[9].cuda().squeeze(), sample[10].cuda().squeeze(), sample[11].squeeze().item()

                #print(dset_train.buildings[building_index]._id)
                #print('num edges', len(edge_gt))

                #images = torch.cat([im_arr.unsqueeze(0).repeat((len(edge_images), 1, 1, 1)), edge_images.unsqueeze(1)], dim=1)
                images = im_arr.unsqueeze(0)
                edge_image_pred, results = model(images, corners, edges, corner_edge_pairs, edge_corner)
                edge_image_gt = edge_images[edge_gt > 0.5].max(0, keepdim=True)[0].unsqueeze(0)
                losses = []
                losses.append(F.binary_cross_entropy(edge_image_pred, edge_image_gt))
                
                loop_gts = []
                multi_loop_gts = []                
                
                for index, result in enumerate(results):
                    edge_pred = result[0]
                    edge_loss = F.binary_cross_entropy(edge_pred, edge_gt)                    
                    losses.append(edge_loss)

                    loop_pred = result[1]
                    loop_edge = result[2]
                    loop_gt = ((loop_edge * edge_gt).sum(-1) == loop_edge.sum(-1)).float()
                    losses.append(F.binary_cross_entropy(loop_pred, loop_gt))
                    loop_gts.append(loop_gt)

                    multi_loop_pred = result[3]
                    multi_loop_edge = result[4]
                    multi_loop_gt = ((multi_loop_edge * edge_gt).sum(-1) == multi_loop_edge.sum(-1)).float()
                    # if multi_loop_gt.sum() < 0.5:
                    #     multi_loop_gt = (torch.arange(len(multi_loop_edge)).cuda() == (multi_loop_edge * edge_gt).sum(-1).max(0)[1]).float()
                    #     pass

                    #losses.append(F.binary_cross_entropy(result[6], multi_loop_gt))                    
                    if multi_loop_gt.sum() > 1.5:
                        multi_loop_gt = (torch.arange(len(multi_loop_edge)).cuda() == ((multi_loop_edge * edge_gt).sum(-1) * multi_loop_gt).max(0)[1]).float()
                        pass
                    losses.append(F.binary_cross_entropy(multi_loop_pred, multi_loop_gt))

                    multi_loop_gts.append(multi_loop_gt)
                    continue
                
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
                    #print(building._id)
                    #building.update_edges(edge_pred.detach().cpu().numpy() > 0.5)
                    for pred_index, result in enumerate(results):
                        edge_pred = result[0]
                        images, _ = building.visualize(mode='', edge_state=edge_pred.detach().cpu().numpy() > 0.5)
                        cv2.imwrite(options.test_dir + '/' + str(index_offset) + '_edge_' + str(pred_index) + '_pred.png', images[0])
                        continue
                    #building.update_edges(edge_confidence.detach().cpu().numpy() > 0.5)
                    images, _ = building.visualize(mode='', edge_state=edge_gt.detach().cpu().numpy() > 0.5, color=[0, 0, 255])
                    cv2.imwrite(options.test_dir + '/' + str(index_offset) + '_edge_gt.png', images[0])
                    #images, _ = building.visualize(mode='', edge_state=edge_pred.detach().cpu().numpy() > 0.5, color=[0, 0, 255])
                    #cv2.imwrite(options.test_dir + '/' + str(index_offset) + '_edge_pred.png', images[0])                    

                    cv2.imwrite(options.test_dir + '/' + str(index_offset) + '_image_gt.png', (edge_image_gt.squeeze().detach().cpu().numpy() * 255).astype(np.uint8))
                    cv2.imwrite(options.test_dir + '/' + str(index_offset) + '_image_pred.png', (edge_image_pred.squeeze().detach().cpu().numpy() * 255).astype(np.uint8))

                    for pred_index, (result, loop_gt, multi_loop_gt) in enumerate(zip(results, loop_gts, multi_loop_gts)):
                        loop_pred = result[1]
                        loop_edge_mask = result[2]                        
                        multi_loop_pred = result[3]
                        multi_loop_edge_mask = result[4]
                        multi_loop_predictions = result[5]
                        
                        edge_mask = (loop_edge_mask * loop_pred.view((-1, 1))).max(0)[0]
                        images, _ = building.visualize(mode='', edge_state=edge_mask.detach().cpu().numpy() > 0.5, color=[0, 255, 0])
                        cv2.imwrite(options.test_dir + '/' + str(index_offset) + '_loop_' + str(pred_index) + '_pred.png', images[0])
                        if (pred_index == len(results) - 1) or True:
                            edge_mask = (loop_edge_mask * loop_gt.view((-1, 1))).max(0)[0]
                            images, _ = building.visualize(mode='', edge_state=edge_mask.detach().cpu().numpy() > 0.5, color=[0, 128, 0], debug=True)
                            cv2.imwrite(options.test_dir + '/' + str(index_offset) + '_loop_' + str(pred_index) + '_gt.png', images[0])
                            if False:
                                for mask_index, edge_mask in enumerate(loop_edge_mask.detach().cpu().numpy()):
                                    images, _ = building.visualize(mode='', edge_state=edge_mask > 0.5, color=[0, 128, 0])
                                    cv2.imwrite(options.test_dir + '/' + str(index_offset) + '_loop_' + str(pred_index) + '_gt_' + str(mask_index) + '.png', images[0])                                
                                    pass
                                pass
                            pass
                        # edge_mask = (multi_loop_edge_mask * multi_loop_pred.view((-1, 1))).max(0)[0]
                        # images, _ = building.visualize(mode='', edge_state=edge_mask.detach().cpu().numpy() > 0.5, color=[0, 255, 255])
                        # cv2.imwrite(options.test_dir + '/' + str(index_offset) + '_multi_loop_' + str(pred_index) + '_pred.png', images[0])
                        for index, multi_loop_index in enumerate(multi_loop_predictions.detach().cpu().numpy()):
                            edge_mask = multi_loop_edge_mask[multi_loop_index]
                            images, _ = building.visualize(mode='', edge_state=edge_mask.detach().cpu().numpy() > 0.5, color=[0, 0, 128])
                            cv2.imwrite(options.test_dir + '/' + str(index_offset) + '_multi_loop_' + str(pred_index) + '_pred_' + str(index) + '.png', images[0])
                            continue
                        
                        if (pred_index == len(results) - 1) or True:
                            edge_mask = (multi_loop_edge_mask * multi_loop_gt.view((-1, 1))).max(0)[0]
                            images, _ = building.visualize(mode='', edge_state=edge_mask.detach().cpu().numpy() > 0.5, color=[0, 128, 128])
                            cv2.imwrite(options.test_dir + '/' + str(index_offset) + '_multi_loop_' + str(pred_index) + '_gt.png', images[0])
                            pass                        
                        continue
                    # exit(1)
                    # loop_corners = [loop.detach().cpu().numpy() for loop in loop_corners]
                    # image = building.visualizeLoops(loop_corners, loop_pred.detach().cpu().numpy() > 0.5)
                    # cv2.imwrite(options.test_dir + '/' + str(index_offset) + '_loop_pred.png', image)
                    # image = building.visualizeLoops(loop_corners, loop_gt.detach().cpu().numpy() > 0.5)
                    # cv2.imwrite(options.test_dir + '/' + str(index_offset) + '_loop_gt.png', image)
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

            with torch.no_grad():
                testOneEpoch(options, model, dset_val) 
                pass
            continue
        continue
    return

def testOneEpoch(options, model, dataset, additional_models=[], visualize=False):
    #model.eval()
    
    epoch_losses = []
    ## Same with train_loader but provide progress bar
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)    
    data_iterator = tqdm(data_loader, total=len(dataset))
    statistics = np.zeros(6)
    statistics_per_length = np.zeros((350, 4))
    all_images = []
    row_images = []    
    for sample_index, sample in enumerate(data_iterator):
        im_arr, corner_images, edge_images, corners, edges, corner_gt, edge_gt, corner_edge_pairs, edge_corner, left_edges, right_edges, building_index = sample[0].cuda().squeeze(0), sample[1].cuda().squeeze(0), sample[2].cuda().squeeze(0), sample[3].cuda().squeeze(0), sample[4].cuda().squeeze(0), sample[5].cuda().squeeze(0), sample[6].cuda().squeeze(0), sample[7].cuda().squeeze(), sample[8].cuda().squeeze(0), sample[9].cuda().squeeze(), sample[10].cuda().squeeze(), sample[11].squeeze().item()

        im_arr, corner_images, edge_images, corners, edges, corner_gt, edge_gt, corner_edge_pairs, edge_corner, left_edges, right_edges, building_index = sample[0].cuda().squeeze(0), sample[1].cuda().squeeze(0), sample[2].cuda().squeeze(0), sample[3].cuda().squeeze(0), sample[4].cuda().squeeze(0), sample[5].cuda().squeeze(0), sample[6].cuda().squeeze(0), sample[7].cuda().squeeze(), sample[8].cuda().squeeze(), sample[9].cuda().squeeze(), sample[10].cuda().squeeze(), sample[11].squeeze().item()

        images = im_arr.unsqueeze(0)
        edge_image_pred, results = model(images, corners, edges, corner_edge_pairs, edge_corner)
        edge_image_gt = edge_images[edge_gt > 0.5].max(0, keepdim=True)[0].unsqueeze(0)
        losses = []
        losses.append(F.binary_cross_entropy(edge_image_pred, edge_image_gt))

        loop_gts = []
        multi_loop_gts = []                

        for index, result in enumerate(results):
            edge_pred = result[0]
            edge_loss = F.binary_cross_entropy(edge_pred, edge_gt)                    
            losses.append(edge_loss)

            loop_pred = result[1]
            loop_edge = result[2]
            loop_gt = ((loop_edge * edge_gt).sum(-1) == loop_edge.sum(-1)).float()
            losses.append(F.binary_cross_entropy(loop_pred, loop_gt))
            loop_gts.append(loop_gt)

            multi_loop_pred = result[3]
            multi_loop_edge = result[4]
            multi_loop_gt = ((multi_loop_edge * edge_gt).sum(-1) == multi_loop_edge.sum(-1)).float()
            # if multi_loop_gt.sum() < 0.5:
            #     multi_loop_gt = (torch.arange(len(multi_loop_edge)).cuda() == (multi_loop_edge * edge_gt).sum(-1).max(0)[1]).float()
            #     pass
            if multi_loop_gt.sum() > 1.5:
                multi_loop_gt = (torch.arange(len(multi_loop_edge)).cuda() == ((multi_loop_edge * edge_gt).sum(-1) * multi_loop_gt).max(0)[1]).float()
                pass
            
            losses.append(F.binary_cross_entropy(multi_loop_pred, multi_loop_gt))
            multi_loop_gts.append(multi_loop_gt)
            continue

        loss = sum(losses)        


        # edges = edges.detach().cpu().numpy() * 256.0
        # lengths = np.sqrt((edges[:, 0] - edges[:, 2])**2 + (edges[:, 1] - edges[:, 3])**2)
        # lengths = lengths.astype('int32')

        # edge_pred = edge_pred.detach().cpu().numpy()

        # edge_gt = edge_gt.detach().cpu().numpy() > 0.5
        # statistics[0] += np.logical_and(edge_pred > 0.5, edge_gt == 1).sum()
        # statistics[1] += np.logical_and(edge_pred < 0.5, edge_gt == 1).sum()                  
        # statistics[2] += np.logical_and(edge_pred > 0.5, edge_gt == 0).sum()                  
        # statistics[3] += np.logical_and(edge_pred < 0.5, edge_gt == 0).sum()
        # statistics[4] += np.all(edge_pred == edge_gt)
        # statistics[5] += 1        
        # statistics_per_length[lengths, 0] += np.logical_and(edge_pred > 0.5, edge_gt == 1).sum()
        # statistics_per_length[lengths, 1] += np.logical_and(edge_pred < 0.5, edge_gt == 1).sum()                  
        # statistics_per_length[lengths, 2] += np.logical_and(edge_pred > 0.5, edge_gt == 0).sum()                  
        # statistics_per_length[lengths, 3] += np.logical_and(edge_pred < 0.5, edge_gt == 0).sum()

        if sample_index % 500 < 16 or visualize:
            index_offset = sample_index % 1000
            building = dataset.buildings[building_index]
            #print(building._id)            
            #building.update_edges(edge_pred.detach().cpu().numpy() > 0.5)
            for pred_index, result in enumerate(results):
                edge_pred = result[0]
                images, _ = building.visualize(mode='', edge_state=edge_pred.detach().cpu().numpy() > 0.5)
                if sample_index % 500 < 16:
                    cv2.imwrite(options.test_dir + '/val_' + str(index_offset) + '_edge_' + str(pred_index) + '_pred.png', images[0])
                    pass
                row_images.append(images[0])                
                continue
            #building.update_edges(edge_confidence.detach().cpu().numpy() > 0.5)
            for pred_index, (result, loop_gt, multi_loop_gt) in enumerate(zip(results, loop_gts, multi_loop_gts)):
                loop_pred = result[1]
                loop_edge_mask = result[2]                        
                multi_loop_pred = result[3]
                multi_loop_edge_mask = result[4]
                multi_loop_predictions = result[5]
            
                edge_mask = (loop_edge_mask * loop_pred.view((-1, 1))).max(0)[0]
                images, _ = building.visualize(mode='', edge_state=edge_mask.detach().cpu().numpy() > 0.5, color=[0, 255, 0])
                if sample_index % 500 < 16:
                    cv2.imwrite(options.test_dir + '/val_' + str(index_offset) + '_loop_' + str(pred_index) + '_pred.png', images[0])
                    pass
                row_images.append(images[0])                
                if (pred_index == len(results) - 1) or True:
                    edge_mask = (loop_edge_mask * loop_gt.view((-1, 1))).max(0)[0]
                    images, _ = building.visualize(mode='', edge_state=edge_mask.detach().cpu().numpy() > 0.5, color=[0, 128, 0], debug=True)
                    if sample_index % 500 < 16:
                        cv2.imwrite(options.test_dir + '/val_' + str(index_offset) + '_loop_' + str(pred_index) + '_gt.png', images[0])
                        pass
                    if pred_index == len(results) - 1:
                        row_images.append(images[0])
                        pass
                    
                    if index_offset == 14 and False:
                        #print(torch.cat([torch.stack([edge_pred, edge_gt], dim=-1), edge_corner.float()], dim=-1))                        
                        order = torch.sort(loop_pred, descending=True)[1]
                        for mask_index, edge_mask in enumerate(loop_edge_mask[order].detach().cpu().numpy()):
                            print(mask_index, loop_pred[order[mask_index]])
                            images, _ = building.visualize(mode='', edge_state=edge_mask > 0.5, color=[0, 128, 0])
                            cv2.imwrite(options.test_dir + '/val_' + str(index_offset) + '_loop_' + str(pred_index) + '_gt_' + str(mask_index) + '.png', images[0])                                
                            pass                            
                        for mask_index, edge_mask in enumerate(multi_loop_edge_mask.detach().cpu().numpy()):
                            images, _ = building.visualize(mode='', edge_state=edge_mask > 0.5, color=[0, 128, 0])
                            cv2.imwrite(options.test_dir + '/val_' + str(index_offset) + '_multi_loop_' + str(pred_index) + '_gt_' + str(mask_index) + '.png', images[0])                                
                            pass
                        exit(1)
                        pass                    
                    pass
                for index, multi_loop_index in enumerate(multi_loop_predictions.detach().cpu().numpy()):
                    edge_mask = multi_loop_edge_mask[multi_loop_index]
                    images, _ = building.visualize(mode='', edge_state=edge_mask.detach().cpu().numpy() > 0.5, color=[0, 0, 128])
                    if sample_index % 500 < 16:                
                        cv2.imwrite(options.test_dir + '/val_' + str(index_offset) + '_multi_loop_' + str(pred_index) + '_pred_' + str(index) + '.png', images[0])
                        pass
                    row_images.append(images[0])
                    continue
                # if (pred_index == len(results) - 1) or True:
                #     edge_mask = (multi_loop_edge_mask * multi_loop_gt.view((-1, 1))).max(0)[0]
                #     images, _ = building.visualize(mode='', edge_state=edge_mask.detach().cpu().numpy() > 0.5, color=[0, 128, 128])
                #     if sample_index % 500 < 16:                
                #         cv2.imwrite(options.test_dir + '/val_' + str(index_offset) + '_multi_loop_' + str(pred_index) + '_gt.png', images[0])
                #         pass
                #     pass                        
                #     if pred_index == len(results) - 1:
                #         row_images.append(images[0])
                #         pass
                #     pass
                continue

            images, _ = building.visualize(mode='', edge_state=edge_gt.detach().cpu().numpy() > 0.5, color=[0, 0, 255])
            row_images.append(images[0])            
            if sample_index % 500 < 16:
                cv2.imwrite(options.test_dir + '/val_' + str(index_offset) + '_edge_gt.png', images[0])
                cv2.imwrite(options.test_dir + '/val_' + str(index_offset) + '_image_gt.png', (edge_image_gt.squeeze().detach().cpu().numpy() * 255).astype(np.uint8))
                cv2.imwrite(options.test_dir + '/val_' + str(index_offset) + '_image_pred.png', (edge_image_pred.squeeze().detach().cpu().numpy() * 255).astype(np.uint8))
                pass
            
            all_images.append(row_images)
            row_images = []
            pass
        continue
    
    if visualize:
        if len(row_images) > 0:
            all_images.append(row_images)
            pass        
        image = tileImages(all_images, background_color=0)
        cv2.imwrite(options.test_dir + '/results.png', image)        
        pass

    print('statistics', statistics[0] / (statistics[0] + statistics[1]), statistics[3] / (statistics[2] + statistics[3]), statistics[4] / statistics[5], statistics[5])
    # for i in range(350):
    #     if (statistics_per_length[i, 0] + statistics_per_length[i, 1] > 0) and (statistics_per_length[i, 2] + statistics_per_length[i, 3] > 0): 
    #         print('{}, {}, {}'.format(i+1, statistics_per_length[i, 0] / (statistics_per_length[i, 0] + statistics_per_length[i, 1] + 1E-10), statistics_per_length[i, 3] / (statistics_per_length[i, 2] + statistics_per_length[i, 3] + 1E-10)))

    print('validation loss', np.array(epoch_losses).mean(0))
    model.train()
    return


def visualizeExample(options, im, label_pred, label_gt, confidence=1, prefix='', suffix=''):
    image = (im[:3] * 255).transpose((1, 2, 0)).astype(np.uint8)
    image[im[3] > 0.5] = np.array([255, 0, 0], dtype=np.uint8)
    left_image = image.copy()
    left_image[im[4] > 0.5] = np.array([0, 0, 255])
    right_image = image.copy()
    right_image[im[5] > 0.5] = np.array([0, 0, 255])
    if label_pred == 1:
        images = [left_image, right_image]
    else:
        images = [right_image, left_image]
        pass
    if label_pred == label_gt:
        #background_color = np.array([0, 0, 0], dtype=np.uint8)
        background_color = int(round(confidence * 255))
    else:
        background_color = np.array([0, 0, int(round(255 * confidence))], dtype=np.uint8)
        pass
    image = tileImages([images], background_color=background_color)
    cv2.imwrite(options.test_dir + '/' + prefix + 'image' + suffix + '.png', image)
    return


if __name__ == '__main__':
    args = parse_args()
    args.keyname = 'edge'
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

    main(args)
