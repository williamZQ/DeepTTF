# encoding utf-8
'''
@Author: william
@Description:
@time:2021/6/5 19:17
'''
import torch
import torch.nn as nn

if __name__ == "__main__":
    conv1 = nn.Conv1d(in_channels=256,out_channels = 100, kernel_size = 2)
    input = torch.randn(32, 35, 256)
    # batch_size x text_len x embedding_size -> batch_size x embedding_size x text_len
    input = input.permute(0, 2, 1)
    out = conv1(input)
    print(out.size())
