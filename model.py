import torch

class GRU(torch.nn.Module):
    def __init__(self,is_train,input_size=181,hidden_size=128):
        super(GRU, self).__init__()
        self.gru=torch.nn.GRU(input_size=input_size,hidden_size=hidden_size,batch_first=True)
        self.fc1=torch.nn.Linear(hidden_size,int(hidden_size/2))
        self.fc2=torch.nn.Linear(int(hidden_size/2),int(hidden_size/4))
        self.fc3=torch.nn.Linear(int(hidden_size/4),1)
        self.loss=torch.nn.MSELoss(reduction="sum")

        self.is_train=is_train
        self.initialize()

    def initialize(self):
        for name,parameter in self.named_parameters():
            if("weight" in name):
                torch.nn.init.normal_(parameter,mean=0,std=0.01)
            if("bias" in name):
                torch.nn.init.constant_(parameter,0.0)

    def forward(self,input,label):
        hn_output=self.gru(input)[0]

        output=self.fc1(hn_output)
        output.relu_()
        output=self.fc2(output)
        output.relu_()
        output=self.fc3(output)
        if(self.is_train):
            loss=self.loss(output.flatten(1),label)
            return loss,output

        else:
            return output


