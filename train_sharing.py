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
from model.loop import LoopModel
from model.modules import findLoopsModule

def main(options):
    ##############################################################################################################
    ############################################### Define Model #################################################
    ##############################################################################################################

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
    with open('{}/building_reconstruction/la_dataset_new/train_list_prime.txt'.format(PREFIX)) as f:
        file_list = [line.strip() for line in f.readlines()]
        train_list = file_list[:-50]
        valid_list = file_list[-50:]
        #valid_list = file_list[:-50]
        pass

    # with open('{}/building_reconstruction/la_dataset_new/valid_list.txt'.format(PREFIX)) as f:
    #     valid_list = [line.strip() for line in f.readlines()]

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

        for epoch in range(10):
            #os.system('rm ' + options.test_dir + '/' + str(num_edges) + '_*')
            dset_train.reset()
            train_loader = DataLoader(dset_train, batch_size=1, shuffle=True, num_workers=1)    
            epoch_losses = []
            ## Same with train_loader but provide progress bar
            data_iterator = tqdm(train_loader, total=len(dset_train))
            optimizer.zero_grad()        
            for sample_index, sample in enumerate(data_iterator):

                im_arr, corner_images, edge_images, corners, edges, corner_gt, edge_gt, corner_edge_pairs, edge_corner, left_edges, right_edges, building_index = sample[0].cuda().squeeze(0), sample[1].cuda().squeeze(0), sample[2].cuda().squeeze(0), sample[3].cuda().squeeze(0), sample[4].cuda().squeeze(0), sample[5].cuda().squeeze(0), sample[6].cuda().squeeze(0), sample[7].cuda().squeeze(), sample[8].cuda().squeeze(), sample[9].cuda().squeeze(), sample[10].cuda().squeeze(), sample[11].squeeze().item()

                #print('num edges', len(edge_gt))

                if len(edge_gt) > 100:
                    continue
                images = torch.cat([im_arr.unsqueeze(0).repeat((len(edge_images), 1, 1, 1)), edge_images.unsqueeze(1)], dim=1)                
                results = model(images, corners, edges, corner_edge_pairs, edge_corner)

                losses = []
                edge_preds = []
                loop_preds = []
                loop_gts = []
                loop_edges = []
                for index, result in enumerate(results):
                    edge_pred = result[0]
                    edge_loss = F.binary_cross_entropy(edge_pred, edge_gt)                    
                    losses.append(edge_loss)
                    edge_preds.append(edge_pred)

                    loop_pred = result[1]
                    loop_edge = result[2]
                    loop_gt = ((loop_edge * edge_gt).sum(-1) == loop_edge_mask.sum(-1)).float()
                    losses.append(F.binary_cross_entropy(loop_pred, loop_gt))
                    loop_preds.append(loop_pred)
                    loop_gts.append(loop_gt)
                    loop_edges.append(loop_edge)
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
                    #building.update_edges(edge_pred.detach().cpu().numpy() > 0.5)
                    for pred_index, edge_pred in enumerate(edge_preds):
                        images, _ = building.visualize(mode='', edge_state=edge_pred.detach().cpu().numpy() > 0.5)
                        cv2.imwrite(options.test_dir + '/' + str(index_offset) + '_image_' + str(pred_index) + '.png', images[0])
                        continue
                    #building.update_edges(edge_confidence.detach().cpu().numpy() > 0.5)
                    images, _ = building.visualize(mode='', edge_state=edge_gt.detach().cpu().numpy() > 0.5)
                    cv2.imwrite(options.test_dir + '/' + str(index_offset) + '_input.png', images[0])

                    for pred_index, (loop_pred, loop_gt, loop_edge_mask) in enumerate(loop_preds, loop_gts, loop_edges):
                        edge_mask = loop_edge_mask * loop_pred.view((-1, 1)).max(0)[0]
                        images, _ = building.visualize(mode='', edge_state=edge_mask.detach().cpu().numpy() > 0.5)
                        cv2.imwrite(options.test_dir + '/' + str(index_offset) + '_loop_' + str(pred_index) + '_pred.png', images[0])
                        edge_mask = loop_edge_mask * loop_gt.view((-1, 1)).max(0)[0]
                        images, _ = building.visualize(mode='', edge_state=edge_mask.detach().cpu().numpy() > 0.5)
                        cv2.imwrite(options.test_dir + '/' + str(index_offset) + '_loop_' + str(pred_index) + '_gt.png', images[0])
                        continue
                    
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

            # with torch.no_grad():
            #     testOneEpoch(options, model, dset_val) 
            #     pass
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
        #edge_confidence = validator.validate(dset_train.buildings[building_index])
        #edges = torch.cat([edges, edge_confidence.unsqueeze(-1)], dim=-1)

        if 'graph' in options.suffix:
            edge_pred, loop_pred, loop_corners, loop_info = model(im_arr.unsqueeze(0), corners, edges, corner_edge_pairs, edge_corner)
        else:
            if len(edge_gt) > 150:
                #all_images.append(np.zeros((256, 256, 3), dtype=np.uint8))
                #all_images.append(np.zeros((256, 256, 3), dtype=np.uint8))                
                continue                
            edge_pred, loop_pred, loop_corners, loop_info = model(im_arr, edge_images, corners, corner_edge_pairs, edge_corner)
            pass
        
        #corner_loss = F.binary_cross_entropy(corner_pred, corner_gt) * 0
        edge_loss = F.binary_cross_entropy(edge_pred, edge_gt)
        losses = [edge_loss]

        num_corners = len(corners)                    
        edge_corner_gt = edge_corner[edge_gt > 0.5]
        corner_pair_mask = torch.zeros(num_corners * num_corners).cuda()
        corner_pair_mask[edge_corner_gt[:, 0] * num_corners + edge_corner_gt[:, 1]] = 1
        corner_pair_mask[edge_corner_gt[:, 1] * num_corners + edge_corner_gt[:, 0]] = 1
        loop_gt = []
        for loop in loop_corners:
            loop_corner_pairs = loop * num_corners + torch.cat([loop[1:], loop[:1]], dim=0)
            loop_gt.append(corner_pair_mask[loop_corner_pairs].min())
            continue
        loop_gt = torch.stack(loop_gt, dim=0)                    
        losses.append(F.binary_cross_entropy(loop_pred, loop_gt))
        
        loss = sum(losses)        

        ## Progress bar
        loss_values = [l.data.item() for l in losses]
        epoch_losses.append(loss_values)
        status = 'val loss: '
        for l in loss_values:
            status += '%0.5f '%l
            continue
        data_iterator.set_description(status)

        edges = edges.detach().cpu().numpy() * 256.0
        lengths = np.sqrt((edges[:, 0] - edges[:, 2])**2 + (edges[:, 1] - edges[:, 3])**2)
        lengths = lengths.astype('int32')

        edge_pred = edge_pred.detach().cpu().numpy()

        edge_gt = edge_gt.detach().cpu().numpy() > 0.5
        statistics[0] += np.logical_and(edge_pred > 0.5, edge_gt == 1).sum()
        statistics[1] += np.logical_and(edge_pred < 0.5, edge_gt == 1).sum()                  
        statistics[2] += np.logical_and(edge_pred > 0.5, edge_gt == 0).sum()                  
        statistics[3] += np.logical_and(edge_pred < 0.5, edge_gt == 0).sum()
        statistics[4] += np.all(edge_pred == edge_gt)
        statistics[5] += 1        
        statistics_per_length[lengths, 0] += np.logical_and(edge_pred > 0.5, edge_gt == 1).sum()
        statistics_per_length[lengths, 1] += np.logical_and(edge_pred < 0.5, edge_gt == 1).sum()                  
        statistics_per_length[lengths, 2] += np.logical_and(edge_pred > 0.5, edge_gt == 0).sum()                  
        statistics_per_length[lengths, 3] += np.logical_and(edge_pred < 0.5, edge_gt == 0).sum()

        if sample_index % 500 < 16 or visualize:
            index_offset = sample_index % 500
            building = dataset.buildings[building_index]

            images, _ = building.visualize(mode='draw_annot', edge_state=edge_gt, building_idx=building_index)                            
            if sample_index % 500 < 16:
                cv2.imwrite(options.test_dir + '/val_' + str(index_offset) + '_input.png', images[0])
                pass
            if sample_index % 500 < 16:
                print(index_offset, building._id)
                np.save(options.test_dir + '/edge_' + building._id + '.png', edge_pred)
                pass
            row_images.append(images[0])
            
            images, _ = building.visualize(mode='draw_annot', edge_state=edge_pred, building_idx=building_index)                            
            if sample_index % 500 < 16:
                cv2.imwrite(options.test_dir + '/val_' + str(index_offset) + '_image.png', images[0])
                pass
            row_images.append(images[0])

            if sample_index == 5 and False:
                image = (im_arr[:3].detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
                print(torch.stack([loop_pred, loop_gt], dim=-1))
                for loop_index, loop_mask in enumerate(loop_info.detach().cpu().numpy()):
                    loop_image = image.copy()
                    loop_image[loop_mask > 0.5] = np.array([255, 0, 0])
                    loop_image[loop_mask < -0.5] = np.array([0, 0, 255])                        
                    #cv2.imwrite(options.test_dir + '/val_' + str(index_offset) + '_mask_' + str(loop_index) + '.png', (loop_mask * 255).astype(np.uint8))
                    cv2.imwrite(options.test_dir + '/val_' + str(index_offset) + '_mask_' + str(loop_index) + '.png', loop_image)
                    continue
                pass

            loop_corners = [loop.detach().cpu().numpy() for loop in loop_corners]
            image = building.visualizeLoops(loop_corners, loop_pred.detach().cpu().numpy() > 0.5)
            if sample_index % 500 < 16:
                cv2.imwrite(options.test_dir + '/val_' + str(index_offset) + '_loop_pred.png', image)
                pass
            row_images.append(image)
            image = building.visualizeLoops(loop_corners, loop_gt.detach().cpu().numpy() > 0.5)
            if sample_index % 500 < 16:
                cv2.imwrite(options.test_dir + '/val_' + str(index_offset) + '_loop_gt.png', image)
                pass
            row_images.append(image)
            
            if visualize:
                #print(building._id)
                #np.save('test/confidence.npy', edge_pred)
                #print(edge_pred[35])
                #edge_pred[35] = 1
                images, _ = building.visualize(mode='draw_annot', edge_state=edge_pred, building_idx=building_index, post_processing=True)                            
                row_images.append(images[0])
                pass

            for additional_model in additional_models:
                edge_pred = additional_model(image_inp, left_edges, right_edges)
                edge_pred = edge_pred.detach().cpu().numpy()
                images, _ = building.visualize(mode='draw_annot', edge_state=edge_pred, building_idx=building_index)
                row_images.append(images[0])
                if visualize:
                    images, _ = building.visualize(mode='draw_annot', edge_state=edge_pred, building_idx=building_index, post_processing=True)
                    row_images.append(images[0])
                    pass
                continue
            #row_images.append(images[1])
            #row_images.append(images[2]) 
            if len(row_images) >= 10:
                all_images.append(row_images)
                row_images = []
                pass
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
    args.keyname = 'sharing'
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
