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
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.m = data.m
        self.w = data.w

        self.nodes_num_list = [self.m] * numlayers
        self.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.last_layer = self.m
        self.pool_list = [-1] * numlayers

        self.cfgs = [32] * numlayers

        self.L = Variable(data.L_cheby).cuda()
        x = self.w
        self.linears, self.linears2, self.linears3, self.linears4, self.batch_norm = [], [], [], [], [];
        for i in range(len(self.cfgs)):
            self.linears += [nn.Linear(x, self.cfgs[i])];
            self.linears2 += [nn.Linear(x, self.cfgs[i])]
            self.linears3 += [nn.Linear(x, self.cfgs[i])]
            self.linears4 += [nn.Linear(x, self.cfgs[i])]
            self.batch_norm += [nn.BatchNorm1d(self.cfgs[i])];
            x = self.cfgs[i]
            
        self.linears = nn.ModuleList(self.linears);
        self.linears2 = nn.ModuleList(self.linears2);
        self.linears3 = nn.ModuleList(self.linears3);
        self.linears4 = nn.ModuleList(self.linears4);
        self.batch_norm = nn.ModuleList(self.batch_norm)
        self.output = None;
        
        
        self.fc1 = nn.Linear(self.cfgs[-1] * self.m, 256);
        self.fc2 = nn.Linear(256, 53)
        self.dropout = nn.Dropout(args.dropout);
        
    
    # def linear_layer(self, i, x, L, L2, L3, batch_size):
    #     in_feature = x.size(2);
    #     Lx = torch.bmm(L,x);
    #     m = self.nodes_num_list[i];
    #     Lx = Lx.view(batch_size * m, in_feature);
    #     Lx = self.linears[i](Lx);
    #     L2x = torch.bmm(L2,x);
    #     L2x = L2x.view(batch_size * m, in_feature);
    #     L2x = self.linears3[i](L2x);
    #     L3x = torch.bmm(L3,x);
    #     L3x = L3x.view(batch_size * m, in_feature);
    #     L3x = self.linears4[i](L3x);        
    #     x = x.view(batch_size * m, in_feature);
    #     x = self.linears2[i](x);
    #     x = x + Lx + L3x + L2x;
    #     x = x.view(batch_size, m, -1);
    #     return x;

    def linear_layer(self, i, x, L, batch_size):
        in_feature = x.size(2)
        x = x.permute(1,0,2).contiguous()
        x0 = x.view(self.nodes_num_list[i], -1)
        x1 = torch.spmm(L,x0)
        x2 = torch.spmm(L,x1)-x0
        x3 = torch.spmm(L,x2)-x1
        x0 = x0.view(batch_size * self.nodes_num_list[i], in_feature)
        x1 = x1.view(batch_size * self.nodes_num_list[i], in_feature)
        x2 = x2.view(batch_size * self.nodes_num_list[i], in_feature)
        x3 = x3.view(batch_size * self.nodes_num_list[i], in_feature)
        x0 = self.linears[i](x0)
        x1 = self.linears2[i](x1)
        x2 = self.linears3[i](x2)
        x3 = self.linears4[i](x3)
        x = x0 + x1 + x2 + x3
        x = x.view(self.nodes_num_list[i], batch_size, -1)
        x = x.permute(1,0,2).contiguous()
        return x;
        
    def forward(self, inputs):
        x = inputs
        L = self.L
        batch_size = x.size(0)
        for i in range(len(self.cfgs)):
            x = self.linear_layer(i, x, L, batch_size);
            x = x.permute(0,2,1).contiguous();
            x = self.batch_norm[i](x);
            x = x.permute(0,2,1).contiguous();
            x = F.relu(x);
            #x = self.dropout(x);
        x = x.view(batch_size, self.m * self.cfgs[-1]);
        x = F.relu(self.fc1(x))
        x = self.dropout(x);
        x = self.fc2(x)
        return x
    