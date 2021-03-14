import torch
import torch.nn as nn
import torch.nn.functional as F

class CHOWDER(torch.nn.Module):
        def __init__(self, input_size=2048, R=5,neurons=[200,100],p=0.1,lymph_count = False,num_add_features=0):
            super(CHOWDER, self).__init__()
            self.input_size = input_size
            self.R = R
            self.neurons  = neurons
            self.p = p
            self.lymph_count = lymph_count
            self.num_add_features = num_add_features*lymph_count

            self.conv1d = nn.Conv1d(self.input_size,1,1)
            self.fc1 = nn.Linear(self.R*2+self.num_add_features, self.neurons[0]) # a modifier en fonction du nb de features
            self.fc2 = nn.Linear(self.neurons[0], self.neurons[1])
            self.fc_out = nn.Linear(self.neurons[1], 1)
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(self.p)

        def forward(self, in_features,add_features):
            aggregated_features =self.conv1d(in_features)
            top_features = aggregated_features.topk(self.R)[0]
            neg_evidence = aggregated_features.topk(self.R,largest=False)[0]
            MIL_features = torch.cat((top_features,neg_evidence),dim=2)

            if self.lymph_count:
                features_lymp=add_features.reshape(-1,1,self.num_add_features)
                MIL_features = torch.cat((MIL_features,features_lymp),dim=2)

            x = self.fc1(MIL_features)
            x = self.sigmoid(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            x = self.dropout(x)
            out = self.sigmoid(self.fc_out(x))
            return out

class DeepMIL(torch.nn.Module):
        def __init__(self, input_size=2048, attention=198,neurons=64,p=0.1,lymph_count = False,num_add_features=0):
            super(DeepMIL, self).__init__()
            self.input_size = input_size
            self.attention = attention
            self.neurons = neurons
            self.p = p
            self.lymph_count = lymph_count
            self.num_add_features = num_add_features*lymph_count

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

            self.fc2 = nn.Linear(self.attention+self.num_add_features,self.neurons)

            self.fc_out = nn.Linear(self.neurons,1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(self.p)

        def forward(self, in_features,add_features):
            MIL_features = self.fc1(in_features.transpose(-1,1))
            A_V = self.attention_V(MIL_features)
            A_U = self.attention_U(MIL_features)
            A = self.attention_weights(A_V * A_U)
            A = torch.transpose(A, 1, -1)
            A = F.softmax(A, dim=-1)
            M = torch.matmul(A, MIL_features)
            if self.lymph_count:
                features_lymp=add_features.reshape(-1,1,self.num_add_features)
                M = torch.cat((M,features_lymp),dim=2)
            x = self.fc2(M)
            x = self.relu(x)
            x = self.dropout(x)
            out = self.sigmoid(self.fc_out(x))
            return out



class auto_DeepMIL(torch.nn.Module):
        def __init__(self, input_size=2048, attention=128,neurons=64,p=0.1,lymph_count = False,num_add_features=0):
            super(auto_DeepMIL, self).__init__()
            self.input_size = input_size
            self.attention = attention
            self.neurons = neurons
            self.p = p
            self.lymph_count = lymph_count
            self.num_add_features = num_add_features*lymph_count
            self.features_space_size=attention ### embbeding size

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

            self.fc2 = nn.Linear(self.attention+self.num_add_features,self.neurons)

            # ============= VAE part: start ================ #
            self.encoder =torch.nn.Sequential(
                 nn.Linear(self.input_size,1024),
                 nn.Sigmoid(),
                 nn.Linear(1024,512 ),
                 nn.Sigmoid()
            )

            self.get_mu=torch.nn.Sequential(
                nn.Linear(512, self.attention)
            )
            self.get_logvar = torch.nn.Sequential(
                nn.Linear(512, self.features_space_size)
            )
            self.get_temp=torch.nn.Sequential(
                nn.Linear(self.features_space_size, self.attention*self.features_space_size )
            )
            # ============= VAE part: stop ================ #

            self.fc_out = nn.Linear(self.neurons,1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(self.p)

        def get_z(self,mu,logvar):
            eps=torch.randn(198,self.attention)
            eps=torch.autograd.Variable(eps)
            z=mu+eps*torch.exp(logvar/2)
            return z

        def forward(self, in_features,add_features):
            # VAE part
            out1=self.encoder(in_features.transpose(-1,1))
            out2=self.encoder(in_features.transpose(-1,1))

            mu=self.get_mu(out1)
            logvar=self.get_logvar(out2)

            z=self.get_z(mu,logvar)

            # attention mecanism
            MIL_features = self.fc1(in_features.transpose(-1,1))
            A_V = self.attention_V(MIL_features)
            A_U = self.attention_U(MIL_features)
            A = self.attention_weights(A_V * A_U)
            A = torch.transpose(A, 1, -1)
            A = F.softmax(A, dim=-1)

            # multiplication with new sampled features
            M = torch.matmul(A, z)

            if self.lymph_count:
                features_lymp=add_features.reshape(-1,1,self.num_add_features)
                M = torch.cat((M,features_lymp),dim=2)

            x = self.fc2(M)
            x = self.relu(x)
            x = self.dropout(x)
            out = self.sigmoid(self.fc_out(x))
            return out,z, mu, logvar
