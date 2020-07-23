import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from PoolingLayer import Pooling

# cfgs = [64] * 5
# L_cfgs = [0] * 5;
# nodes_num_list = [784] * 5
# pool_list = [-1] * 5
# last_layer = 784
numlayers = 2

class Model(nn.Module):
    def __init__(self, args, data,ex_index):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.m = data.m
        self.w = data.w
        numlayers = args.numlayers
        self.nodes_num_list = [self.m] * numlayers
        self.last_layer = self.m
        self.pool_list = [-1] * numlayers
        self.L = Variable(data.L_gcn).cuda()
        self.cfgs = [32] * numlayers

        self.extra_input_size = len(ex_index)

        x = self.w
        self.linears, self.batch_norm = [], []
        for i in range(len(self.cfgs)):
            self.linears += [nn.Linear(x, self.cfgs[i])]
            self.batch_norm += [nn.BatchNorm1d(self.cfgs[i])];
            x = self.cfgs[i]
        self.linears = nn.ModuleList(self.linears)
        self.batch_norm = nn.ModuleList(self.batch_norm)
        self.output = None
        
        
        self.fc1 = nn.Linear(self.cfgs[-1] * self.m + self.extra_input_size, 512);
        self.fc2 = nn.Linear(512, 2);
        self.dropout = nn.Dropout(args.dropout);
        
    
    # def linear_layer(self, i, x, L, batch_size):
    #     in_feature = x.size(2);
    #     Lx = torch.bmm(L,x);
    #     m = self.nodes_num_list[i];
    #     Lx = Lx.view(batch_size * m, in_feature);
    #     Lx = self.linears[i](Lx);      
    #     x = x.view(batch_size * m, in_feature);
    #     x = self.linears2[i](x);
    #     x = x+Lx
    #     x = x.view(batch_size, m, -1);
    #     return x;
    def linear_layer(self, i, x, L, batch_size):
        in_feature = x.size(2)
        # contiguous : 함수 결과가 실제로 메모리에도 우리가 기대하는 순서로 유지하는 역할
        # permute : 모든 차원의 순서 재배치
        x = x.permute(1,0,2).contiguous()  # 2->1, 0->0, 1->2 로 차원 변경
        # view : tensor의 차원 변경
        x = x.view(self.nodes_num_list[i], -1)
        Lx = torch.spmm(L,x)
        Lx = Lx.view(batch_size * self.nodes_num_list[i], in_feature);
        Lx = self.linears[i](Lx);
        x = Lx
        x = x.view(self.nodes_num_list[i], batch_size, -1)
        x = x.permute(1,0,2).contiguous()
        return x

    # outputs = model(in1,in2) 로 forward 호출
    def forward(self, inputs,ex_inputs):
        x = inputs
        L = self.L
        batch_size = x.size(0);
        for i in range(len(self.cfgs)):
            x = self.linear_layer(i, x, L, batch_size);
            x = x.permute(0,2,1).contiguous();
            x = self.batch_norm[i](x);
            x = x.permute(0,2,1).contiguous();
            x = F.relu(x);

            #x = self.dropout(x);
        x = x.view(batch_size, self.m * self.cfgs[-1]);

        # print(x.shape,ex_inputs.shape)
        x = F.relu(self.fc1(torch.cat((x,ex_inputs),1)))
        x = self.dropout(x);
        x = self.fc2(x)
        return x
    