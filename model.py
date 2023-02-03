import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import pytorch_ssim
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as chd
import ChamferDistancePytorch.fscore as fscore


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
            )
        
        self.fc = nn.Sequential(
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256,9)
            )
        
    def forward(self,x):
        batchsize = x.size()[0]
        x=self.conv(x)
        x=torch.max(x,2,keepdim=True)[0]
        x=x.view(-1,1024)
        
        x=self.fc(x)
        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x=x+iden
        x=x.view(-1,3,3)
        return x
        
class STNkd(nn.Module):
    def __init__(self,k=64):
        super(STNkd, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(k,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
            )
        
        self.fc = nn.Sequential(
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256,k*k)
            )
        self.k = k
        
    def forward(self,x):
        batchsize = x.size()[0]
        x=self.conv(x)
        x=torch.max(x,2,keepdim=True)[0]
        x=x.view(-1,1024)
        
        x=self.fc(x)
        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x=x+iden
        x=x.view(-1,self.k,self.k)
        return x
    
class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True,feature_transform = False):
        super(PointNetfeat,self).__init__()
        self.stn = STN3d()
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        self.conv1 = nn.Sequential(
            nn.Conv1d(3,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024)
            )
    def forward(self,x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x,trans)
        x = x.transpose(2,1)
        
        x=self.conv1(x)
        
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x,trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None
            
        pointfeat = x
        x = self.conv2(x)
        x=torch.max(x,2,keepdim=True)[0]
        x=x.view(-1,1024)
        
        if self.global_feat:
            return x, trans,trans_feat
        else:
            x=x.view(-1,1024,1).repeat(1,1,n_pts)
            return torch.cat([x,pointfeat],1),trans,trans_feat
        
class PointNet_encoder(nn.Module):
    def __init__(self):
        super(PointNet_encoder,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(3,64,kernel_size=1 , stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 128, kernel_size=1 , stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 512, kernel_size=1 , stride=1),
            nn.BatchNorm1d(512)
            )

        self.conv2 = nn.Sequential(
            nn.Conv1d(15000,2500,1),
            nn.ReLU(),
            
            nn.Conv1d(2500,256,1),
            nn.ReLU(),
            
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            )
        self.fc = nn.Sequential(
            nn.Linear(32*512, 256),
            nn.ReLU(),
            nn.Linear(256,16),
            )
        
        self.pool = nn.AdaptiveAvgPool1d(15000)
        
    def forward(self,x):
        
        x = self.pool(x)
        
        x = self.conv1(x)
        
        x = x.transpose(2,1)
        x = self.conv2(x)

        #x = x.view(-1,512*32)
        #x = self.fc(x)
        
        
        return  x

class PointNet_decoder(nn.Module):
    def __init__(self):
        super(PointNet_decoder,self).__init__()

        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(512,128,kernel_size=1 , stride=1),
            nn.BatchNorm1d(128),  
            nn.ReLU(),
            
            nn.ConvTranspose1d(128,64,kernel_size=1 , stride=1),
            nn.BatchNorm1d(64),  
            nn.ReLU(),
            
            nn.ConvTranspose1d(64,3,kernel_size=1 , stride=1),

            )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose1d(128,256,1),
            nn.ReLU(),
            
            nn.ConvTranspose1d(256, 2500, 1),
            nn.ReLU(),
            
            nn.ConvTranspose1d(2500, 15000, 1),
            )
        self.fc = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256,512*32),
            )
        
        
        
    def forward(self,x):
        
        #x = self.fc(x)
        #x = x.view(-1,512,32)
        x = self.conv2(x)
        x = x.transpose(2,1)
        
        x = self.conv1(x)
       
        
        return x

#from https://github.com/TropComplique/point-cloud-autoencoder/blob/master/loss.py
class ChamferDistance(nn.Module):

    def __init__(self):
        super(ChamferDistance, self).__init__()
    def min_dist(self, pt_0, set_of_pt):
        dist = None
        size = set_of_pt.shape[1]
        for i in range(size):
            pt_1 = set_of_pt[:,i]
            #print('pt_0 = ',pt_0, 'pt_1 = ',pt_1, 'minus is',pt_0 -pt_1 )
            if dist == None:
                dist = torch.sum(torch.sqrt(torch.pow(pt_0 - pt_1,2)))
            else:
                dist = min(dist,torch.sum(torch.sqrt(torch.pow(pt_0 - pt_1,2))))
                print('dist =',dist)
        return dist
    def forward(self, x, y):
        """
        The inputs are sets of d-dimensional points:
        x = {x_1, ..., x_n} and y = {y_1, ..., y_m}.
        Arguments:
            x: a float tensor with shape [b, d, n].
            y: a float tensor with shape [b, d, m].
        Returns:
            a float tensor with shape [].
        """
        
        x = x.unsqueeze(3)  # shape [b, d, n, 1]
        y = y.unsqueeze(2)  # shape [b, d, 1, m]

        # compute pairwise l2-squared distances
        d = torch.pow(x - y, 2)  # shape [b, d, n, m]
        d = d.sum(1)  # shape [b, n, m]

        min_for_each_x_i, _ = d.min(dim=2)  # shape [b, n]
        min_for_each_y_j, _ = d.min(dim=1)  # shape [b, m]

        distance = min_for_each_x_i.sum(1) + min_for_each_y_j.sum(1)  # shape [b]
        return distance.mean(0)
        
        
        #modified
        '''
        b_size = x.shape[0]
        n = x.shape[2]
        m = y.shape[2]
        dist = 0
        for i in range(b_size):
            for a in range(n):
                dist += self.min_dist(x[i,:,a],y[i])/n
            print('dist =',dist)
            for b in range(m):
                dist += self.min_dist(y[i,:,b],x[i])/m
            print('dist =',dist)
        return dist
        '''
    
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = PointNet_encoder()
        self.decoder = PointNet_decoder()
        self.loss =ChamferDistance()
    
    def forward(self,x):
        
        x = self.encoder(x)
        #print('encode feat = ',x.size())
        x = self.decoder(x)
        
        return x
    
    def get_loss(self,pred,x):
        
        epsilon=1e-20
        
        m = nn.AdaptiveAvgPool1d(15000)
        #loss = self.loss(pred, x)
        x = m(x)
        loss = nn.functional.mse_loss(pred, x)
        
        '''
        x = x.transpose(2,1)
        pred = pred.transpose(2,1)
        
        chamLoss = chd.chamfer_3DDist()
        dist1, dist2, idx1, idx2 = chamLoss(pred, x)
        '''
        #f_score, precision, recall, dist = fscore.fscore(dist1, dist2)
        #f_score.requires_grad= True
        #print('dist  = ', dist)
        #loss = dist
        #loss =  -torch.log(torch.sum(f_score)+epsilon)

        '''
        loss = self.loss(x,pred)
        print('dist = ',loss)
        '''
        
        return loss
        

    
if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn',out.size())
    
    sim_data_64d = Variable(torch.rand(32,64,2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d',out.size())
    
    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat = ',out.size())
    
    
    encode = PointNet_encoder()
    decode = PointNet_decoder()
    ft = encode(sim_data)
    out = decode(ft)
    print('encode feat = ',ft.size())
    print('decode feat = ', out.size())
    
    auto = autoencoder()
    ft = auto(sim_data)
