import torch
import torch.nn as nn
import torch.nn.functional as F

class CHOWDER(torch.nn.Module):
        def __init__(self, input_size=2048, R=5,neurons=[200,100],p=0.5):
            super(CHOWDER, self).__init__()
            self.input_size = input_size
            self.R = R
            self.neurons  = neurons
            self.p = p
            self.conv1d = nn.Conv1d(self.input_size,1,1)
            self.fc1 = nn.Linear(self.R*2, self.neurons[0])
            self.fc2 = nn.Linear(self.neurons[0], self.neurons[1])
            self.fc_out = nn.Linear(self.neurons[1], 1)
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(self.p)

        def forward(self, in_features):
            aggregated_features =self.conv1d(in_features)
            top_features = aggregated_features.topk(self.R)[0]
            neg_evidence = aggregated_features.topk(self.R,largest=False)[0]
            MIL_features = torch.cat((top_features,neg_evidence),dim=2)
            x = self.fc1(MIL_features)
            x = self.sigmoid(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            x = self.dropout(x)
            out = self.sigmoid(self.fc_out(x))
            return out

class DeepMIL(torch.nn.Module):
        def __init__(self, input_size=2048, attention=128,neurons=64,p=0.5):
            super(DeepMIL, self).__init__()
            self.input_size = input_size
            self.attention = attention
            self.neurons = neurons
            self.p = p
            self.fc1 = nn.Linear(self.input_size,self.attention)
            self.attention_V = nn.Sequential(
                nn.Linear(self.attention, self.input_size),
                nn.Tanh()
            )
            self.attention_U = nn.Sequential(
                nn.Linear(self.attention, self.input_size),
                nn.Sigmoid()
            )
            self.attention_weights = nn.Linear(self.input_size,1)
            self.fc2 = nn.Linear(self.attention,self.neurons)
            self.fc_out = nn.Linear(self.neurons,1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(self.p)

        def forward(self, in_features):
            MIL_features = self.fc1(in_features.transpose(-1,1))
            A_V = self.attention_V(MIL_features)
            A_U = self.attention_U(MIL_features)
            A = self.attention_weights(A_V * A_U)
            A = torch.transpose(A, 1, -1)
            A = F.softmax(A, dim=-1)  
            M = torch.matmul(A, MIL_features)
            x = self.fc2(M)
            x = self.relu(x)
            x = self.dropout(x)
            out = self.sigmoid(self.fc_out(x))
            return out


    
