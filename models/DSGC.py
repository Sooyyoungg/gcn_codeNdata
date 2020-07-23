import torch
import torch.nn as nn
import torch.nn.functional as F
from models.XeLayer import XeLayer
# from PoolingLayer import Pooling
from torch.nn.parameter import Parameter
from torch.autograd import Variable

# cfgs = [64] * 5
#L_cfgs = [0] * 5;
# nodes_num_list = [1000] * 5;
# nn_num_list = [64] * 5;
# pool_list = [-1] * 5

last_layer = 1000;

class Model(nn.Module):
    def __init__(self, args, data,ex_index):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.w = data.w
        self.m = data.m
        # self.embed = args.embed;
        self.embed_dim = data.m #data.embed_dim

        numlayers = args.numlayers

        self.L_idx = Variable(data.L_idx).cuda()
        self.pseudo = Variable(data.pseudo).cuda()

        self.input = self.w;
        self.cfgs = [32] * numlayers
        self.nodes_num_list = [self.m] * numlayers
        # self.nn_num_list = [64] * numlayers
        self.graph_num = [1] * numlayers;

        x = self.w #self.input * graph_num[0];
        layers = []
        norm_layers = []
        for i in range(len(self.cfgs)):
            # print(x,self.graph_num[i])
            layers += [XeLayer(self.nodes_num_list[i], x, self.cfgs[i], self.graph_num[i])];
            norm_layers += [nn.BatchNorm1d(self.cfgs[i])];
            x = self.cfgs[i];

        self.extra_input_size = len(ex_index)
        self.fc1 = nn.Linear(self.cfgs[-1] * self.m + self.extra_input_size, 128);
        self.fc2 = nn.Linear(128, 2);
        
        self.linears = nn.ModuleList(layers);
        self.batch_norm = nn.ModuleList(norm_layers);
        
        self.dropout = nn.Dropout(args.dropout);
        
        self.Lhid = 256
        
        self.L_linear1 = nn.ModuleList([nn.Linear(self.embed_dim, self.Lhid) for i in range(sum(self.graph_num))]);
        self.L_linear2 = nn.ModuleList([nn.Linear(self.Lhid, self.embed_dim//2) for i in range(sum(self.graph_num))]);
    
    def reset_parameters(self):
        self.self_embed.data.normal_(0, 1)
    
    def get_L(self, embed, idx, batch_size, layer, graph_id):

        m = self.nodes_num_list[graph_id];
        nn_num = 16 #self.nn_num_list[graph_id];

        # print(embed.shape)#[152, 16, 2]
        # print(idx.shape)#[2432]
        embed = embed.view(-1,self.embed_dim);
        embed = F.tanh(self.L_linear1[layer](embed))
        context = self.L_linear2[layer](embed);

        # print(context.shape,m,nn_num)#[32, 1]) 152 64

        context = context.view(m, nn_num);
        attn = F.softmax(context);
        attn = attn.view(m * nn_num);
        L = Variable(torch.zeros(m * m));
        if self.use_cuda:
            L = Variable(torch.zeros(m * m).cuda());

        L[idx] = attn;
        L = L.view(m, m);
        L = L.expand(batch_size, *L.size());
        return L
    
    def forward(self, inputs,ex_inputs):
        # x = inputs[0];
        # L = inputs[1];
        # batch_size = x.size(0);
        # x = x.transpose(2, 1).contiguous();

        x = inputs;
        # maps_list = inputs[1]; #adj
        # L_list = inputs[2]; #L_idx


        batch_size = x.size(0);
        # x = x.transpose(2,1).contiguous();


        L = []
        s = 0;
        for i in range(len(self.cfgs)):
            L += [[]];
            for j in range(self.graph_num[i]):
                L[i].append(self.get_L(self.pseudo, self.L_idx, batch_size, s + j, i));
            s = s + self.graph_num[i];


        x = torch.cat([x for i in range(self.graph_num[0])],2);
        # print(x.shape)
        for i in range(len(self.cfgs)):
            x = self.linears[i](x, L[i]);
            x = x.permute(0,2,1).contiguous();
            x = self.batch_norm[i](x);
            x = x.permute(0,2,1).contiguous();
            x = F.relu(x);
            x = self.dropout(x);
        x = x.view(batch_size, self.m * self.cfgs[-1]);
        # x = x.view(batch_size, last_layer * self.cfgs[-1]);
        x = F.relu(self.fc1(torch.cat((x, ex_inputs), 1)))
        # x = F.relu(self.fc1(x))
        x = self.dropout(x);
        x = self.fc2(x)
        return x
    
        
        
        