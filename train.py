import torch
import torch.nn as nn
import wandb

from dataset import ModelNetDataLoader
from models.pointnet2 import pointnet
from models.superglue import SuperGlue
from utils import compute_rigid_transform, loss_fn, compute_error


def main():
    torch.backends.cudnn.enabled = False
    USE_WANDB = True
    if USE_WANDB:
        wandb.init(project='group-project')
        
    """Load the dataset"""
    train_dataset_args = {
        'root': '/home/ubuntu/gp/modelnet40/modelnet40_normal_resampled',
        'args': {
            'use_normals': False,
            'num_category': 40,
            'num_point': 200,
            'use_uniform_sample': True,
        },
        'split': 'train',
        'process_data': True,
    }
    train_dataset = ModelNetDataLoader(
        train_dataset_args['root'],
        train_dataset_args['args'],
        train_dataset_args['split'],
        train_dataset_args['process_data']
    )
    
    test_dataset_args = {
        'root': '/home/ubuntu/gp/modelnet40/modelnet40_normal_resampled',
        'args': {
            'use_normals': False,
            'num_category': 40,
            'num_point': 200,
            'use_uniform_sample': True,
        },
        'split': 'test',
        'process_data': True,
    }
    test_dataset = ModelNetDataLoader(
        test_dataset_args['root'],
        test_dataset_args['args'],
        test_dataset_args['split'],
        test_dataset_args['process_data']
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    """Load the model"""
    device = torch.device('cuda')
    pn_model = pointnet()
    pn_model.to(device)
    # Load pre_trained model:
    # https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/log/part_seg/pointnet2_part_seg_msg/checkpoints/best_model.pth
#     pretrained_pn_state_dict = torch.load('/home/ubuntu/gp/models/weights/pointnet_pplus.pth')
#     pn_state_dict = pn_model.state_dict()

#     for name, param in pretrained_pn_state_dict.items():
#         if name not in pn_state_dict:
#             continue
#         elif pn_state_dict[name].shape != param.shape:
#             continue
#         else:
#             pn_state_dict[name] = param

#     pn_model.load_state_dict(pn_state_dict)

#     print('Loaded Pointnet++ model')

    # Initialize the SuperGlue model with weights 'indoor':
    # https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/weights/superglue_indoor.pth
    sg_config = {'descriptor_dim': 256}
    sg_model = SuperGlue(sg_config)
#     sg_state_dict = torch.load('/home/ubuntu/gp/models/weights/superglue_indoor.pth')
#     sg_model.load_state_dict(sg_state_dict)
    sg_model.to(device)
    
    """Specify the traning parameters"""
    lr = 0.0001
    momentum = 0.9
    weight_decay = 0.0005
    optimizer = torch.optim.Adam([
                {'params': pn_model.parameters(), 'lr': 1e-3},
                {'params': sg_model.parameters(), 'lr': 1e-4}
            ], lr=lr)
    
    """Do the training"""
    epoch = 20
    training_loss_visualize_period = 10
    test_period = 50
    test_num_batches = 40
    loss_meter = AverageMeter()
    loss_reproj_meter = AverageMeter()
    loss_inlier_meter = AverageMeter()
    matching_cnt_meter = AverageMeter()
    for e in range(epoch):
        pn_model.train()
        sg_model.train()
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            point_set_1 = data['point_set_1'].float().cuda()
            point_set_2 = data['point_set_2'].float().cuda()
            gt_rotation = data['gt_rotation'].cuda()
            gt_translation = data['gt_translation'].cuda()
            label = data['label']

            output_1 = pn_model(point_set_1.transpose(2,1))
            output_2 = pn_model(point_set_2.transpose(2,1))
            sg_input = {}
            sg_input['keypoints0'] = point_set_1
            sg_input['keypoints1'] = point_set_2
            sg_input['descriptors0'] = output_1
            sg_input['descriptors1'] = output_2
            sg_input['scores0'] = None
            sg_input['scores1'] = None

            result = sg_model(sg_input)
            
            T = compute_rigid_transform(point_set_1, point_set_2, result['matches0'], result['matching_scores0'])

            print("epoch: {}, iteration: {}".format(e, i))
    
            loss, cnt, loss_reproj, loss_inlier = loss_fn(T, gt_rotation, gt_translation, result['matches0'], result['matching_scores0'], point_set_1.transpose(2,1))

            loss_meter.update(loss.item())
            loss_reproj_meter.update(loss_reproj.item())
            loss_inlier_meter.update(loss_inlier)
            matching_cnt_meter.update(cnt)
            if USE_WANDB:
                wandb.log({"train/loss": loss_meter.avg})
                wandb.log({"train/matching_cnt": matching_cnt_meter.avg})
                wandb.log({"train/loss_reproj": loss_reproj_meter.avg})
                wandb.log({"train/loss_inlier": loss_inlier_meter.avg})
                
            loss.backward()
            optimizer.step()
        
            if i % training_loss_visualize_period == 0 and i != 0:
                print("epoch: {}, iteration: {}, train loss: {}".format(e, i, loss_meter.avg))
        
            if i % test_period == 0 and i != 0:
                pn_model.eval()
                sg_model.eval()
                test_loss_meter = AverageMeter()
                ani_rot_error_meter = AverageMeter()
                ani_trans_error_meter = AverageMeter()
                iso_rot_error_meter = AverageMeter()
                iso_trans_error_meter = AverageMeter()
                for j, data in enumerate(test_dataloader):
                    point_set_1 = data['point_set_1'].float().cuda()
                    point_set_2 = data['point_set_2'].float().cuda()
                    gt_rotation = data['gt_rotation'].cuda()
                    gt_translation = data['gt_translation'].cuda()
                    label = data['label']
                    
                    with torch.no_grad():
                        output_1 = pn_model(point_set_1.transpose(2,1))
                        output_2 = pn_model(point_set_2.transpose(2,1))
                        sg_input = {}
                        sg_input['keypoints0'] = point_set_1
                        sg_input['keypoints1'] = point_set_2
                        sg_input['descriptors0'] = output_1
                        sg_input['descriptors1'] = output_2
                        sg_input['scores0'] = None
                        sg_input['scores1'] = None

                        result = sg_model(sg_input)
                        
                        T = compute_rigid_transform(point_set_1, point_set_2, result['matches0'], result['matching_scores0'])

                        if T is None:
                            print("Failed to compute T")
                            continue
                            
                        loss, cnt, _, _ = loss_fn(T, gt_rotation, gt_translation, result['matches0'], result['matching_scores0'], point_set_1.transpose(2,1))
                        test_loss_meter.update(loss.item())
                        
                        ani_rot_error, ani_trans_error, iso_rot_error, iso_trans_error = compute_error(T, gt_rotation, gt_translation)
                        ani_rot_error_meter.update(ani_rot_error)
                        ani_trans_error_meter.update(ani_trans_error)
                        iso_rot_error_meter.update(iso_rot_error)
                        iso_trans_error_meter.update(iso_trans_error)
                    
                    if j >= test_num_batches:
                        break
                
                pn_model.train()
                sg_model.train()
                print("\n Test result:")
                print("epoch: {}, iteration: {}, test loss: {}".format(epoch, i, test_loss_meter.avg))
                print("ani_rot_error: {}, ani_trans_error: {}".format(ani_rot_error_meter.avg, ani_trans_error_meter.avg))
                print("iso_rot_error: {}, iso_trans_error: {}".format(iso_rot_error_meter.avg, iso_trans_error_meter.avg))
                
                if USE_WANDB:
                    wandb.log({"test/loss": test_loss_meter.avg})
                    wandb.log({"test/ani_rot_error": ani_rot_error_meter.avg})
                    wandb.log({"test/ani_trans_error": ani_trans_error_meter.avg})
                    wandb.log({"test/iso_rot_error": iso_rot_error_meter.avg})
                    wandb.log({"test/iso_trans_error": iso_trans_error_meter.avg})

                    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()