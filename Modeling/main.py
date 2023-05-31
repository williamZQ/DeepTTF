# encoding utf-8
'''
@Author: william
@Description:
@time:2021/6/7 18:14
'''

import os
import json
import argparse
import torch
import torch.nn as nn

from Modeling.utils.utils import get_normalized_adj
from Modeling.utils.data_load import Data_load
from Modeling.methods.train import Train
from Modeling.methods.evaluate import Evaluate
import logger

from Modeling.model.TreeCNs import TreeCNs

# from Modeling.model.baselines.GCN import TreeCNs
# from Modeling.model.baselines.GRU import TreeCNs
# from Modeling.model.baselines.OnlyTCN import TreeCNs
# from Modeling.model.baselines.DoubleTCN import TreeCNs

elogger = logger.Logger('run_log')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = json.load(open('./config.json', 'r'))

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--weight_file', type=str, default='./saved_weights/')
parser.add_argument('--timesteps_input', type=int, default=12)
parser.add_argument('--timesteps_output', type=int, default=4)
parser.add_argument('--out_channels', type=int, default=64)
parser.add_argument('--spatial_channels', type=int, default=16)
parser.add_argument('--N', type=int, default=50)
parser.add_argument('--features', type=int, default=1)
parser.add_argument('--time_slice', type=list, default=[1, 2, 3])

parser.add_argument('--MaxNodeNumber', type=int, default=225)
parser.add_argument('--MaxLayerNumber', type=int, default=3)

args = parser.parse_args()

if __name__ == '__main__':
    torch.manual_seed(7)
    NATree, data_set = Data_load(config, args.timesteps_input, args.timesteps_output)
    # NATree = get_normalized_adj(NATree)
    args.N = NATree.shape[0]
    args.MaxNodeNumber = NATree.shape[2]
    args.MaxLayerNumber = NATree.shape[1]
    model = TreeCNs(
        num_nodes=args.N,
        out_channels=args.out_channels,
        spatial_channels=args.spatial_channels,
        features=args.features,
        timesteps_input=args.timesteps_input,
        timesteps_output=args.timesteps_output,
        max_layer_number=args.MaxLayerNumber,
        max_node_number=args.MaxNodeNumber,
    )

    if torch.cuda.is_available():
        model.cuda()
        NATree = torch.from_numpy(NATree).cuda()
        # W_nodes = W_nodes.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 500, 700, 900], gamma=0.5)
    L2 = nn.MSELoss()
    for epoch in range(args.epochs):
        print("Train Process")
        train_loss = Train(
            model=model,
            optimizer=optimizer,
            loss_meathod=L2,
            NATree=NATree,
            data_set=data_set,
            batch_size=args.batch_size
        )
        torch.cuda.empty_cache()
        with torch.no_grad():
            print("Evalution Process")
            eval_loss, eval_index = Evaluate(
                epoch=epoch,
                model=model,
                loss_meathod=L2,
                NATree=NATree,
                time_slice=args.time_slice,
                data_set=data_set
            )
        print("---------------------------------------------------------------------------------------------------")
        print("epoch: {}/{}".format(epoch, args.epochs))
        print("Training loss: {}".format(train_loss))
        for i in range(len(args.time_slice)):
            print("time:{}, Evaluation loss:{}, MAE:{}, RMSE:{}, sMAPE:{}, R2:{}"
                  .format(args.time_slice[i] * 5, eval_loss[-(len(args.time_slice) - i)],
                          eval_index['MAE'][-(len(args.time_slice) - i)],
                          eval_index['RMSE'][-(len(args.time_slice) - i)],
                          eval_index['sMAPE'][-(len(args.time_slice) - i)],
                          eval_index['R-Squared'][-(len(args.time_slice) - i)]))
            elogger.log("time:{}, Evaluation loss:{}, MAE:{}, RMSE:{}, sMAPE:{}, R2:{}"
                        .format(args.time_slice[i] * 5, eval_loss[-(len(args.time_slice) - i)],
                                eval_index['MAE'][-(len(args.time_slice) - i)],
                                eval_index['RMSE'][-(len(args.time_slice) - i)],
                                eval_index['sMAPE'][-(len(args.time_slice) - i)],
                                eval_index['R-Squared'][-(len(args.time_slice) - i)]))
        print("---------------------------------------------------------------------------------------------------")

        if not os.path.exists(args.weight_file):
            os.makedirs(args.weight_file)

        if (epoch % 49 == 0) & (epoch != 0):
            torch.save(model, args.weight_file + 'model_' + str(epoch))
