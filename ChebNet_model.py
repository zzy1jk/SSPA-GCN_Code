import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function

class ChebConv(nn.Module):
    def __init__(self, in_c, out_c, K, bias= True, normalize = False):
        super(ChebConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(K+1 , 1 , in_c , out_c),requires_grad=True)
        # torch.manual_seed(700)
        nn.init.xavier_normal_(self.weight)

        self.Myweight_train = nn.Parameter(torch.Tensor(1,128,in_c) , requires_grad=True) # MODMA

        torch.manual_seed(9)
        nn.init.xavier_normal_(self.Myweight_train)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)
        self.K = K + 1
        self.normalize = normalize

    def forward(self, inputs, graph):
        L = self.get_laplacian(graph, self.normalize)
        mul_L = self.cheb_polynomial(L).unsqueeze(1)
        result1 = torch.matmul(mul_L, inputs)
        result2 = torch.matmul(result1, self.weight)
        result3 = torch.sum(result2, dim=0) + self.bias
        return result3

    def cheb_polynomial(self, laplacian):
        N = laplacian.size(1)  # [N, N]
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device,dtype=torch.float)  # [K, N, N]
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device,dtype=torch.float)
        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1]) - \
                                               multi_order_laplacian[k-2]
        return multi_order_laplacian

    def get_laplacian(self, graph, normalize):
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

class ChebNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, K, chan_num, class_num, dropout):
        super(ChebNet, self).__init__()
        self.class_num = class_num
        self.conv1 = ChebConv(in_c=in_c, out_c=hid_c, K=K)
        self.conv2 = ChebConv(in_c=hid_c, out_c=out_c, K=K)

        self.full_connect1 = nn.Linear(chan_num * out_c, int(chan_num / 4) * out_c)
        self.full_connect2 = nn.Linear(int(chan_num / 4) * out_c, self.class_num)
        self.ReLu = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(int(chan_num / 4) * out_c)


    def forward(self, DE_data, graph_data):
        B = DE_data.size(0)
        out = self.ReLu(self.conv1(DE_data, graph_data))
        # out = self.conv2(out, graph_data)
        output_feature = out.view(B, -1)

        out = self.bn1(self.full_connect1(output_feature))
        out = self.ReLu(out)
        out = self.dropout(out)
        out = self.dropout(self.full_connect2(out))
        class_label_pred = F.softmax(out, dim=1)
        return output_feature, class_label_pred 

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # Store context for backprop
        ctx.alpha = alpha

        # Forward pass is a no-op
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass is just to -alpha the gradient
        output = grad_output.neg() * ctx.alpha

        # Must return same number as inputs to forward()
        return output, None
