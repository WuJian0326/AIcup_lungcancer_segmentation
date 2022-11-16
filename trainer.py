from tqdm import tqdm
from utils_tool import *

torch.backends.cudnn.benchmark = True

class trainer():
    def __init__(self, train_ds, model, optimizer, scheduler, criterion, epochs=500,best_acc=None):
        self.train_ds = train_ds
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.epochs = epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bast_loss = best_acc
        self.scaler = torch.cuda.amp.GradScaler()


    def training(self):
        for idx in range(self.epochs):
            self.train_epoch(idx)
        return self.model


    def train_epoch(self,epo):
        self.model.train()
        total_loss = 0
        #total_batch = 0
        TrainLoader = tqdm(self.train_ds)
        for idx, (image, label) in enumerate(TrainLoader):
            image = image.to(self.device)
            label = label.unsqueeze(1).to(self.device)


            with torch.cuda.amp.autocast():
                output = self.model(image)
                loss = self.criterion(output, label)

            self.optimizer.zero_grad(set_to_none=True)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            TrainLoader.set_description('Epoch ' + str(epo + 1))
            TrainLoader.set_postfix(loss=loss.item(), Learning_rate=self.optimizer.state_dict()['param_groups'][0]['lr'])
            total_loss += loss

        train_loss = total_loss/len(self.train_ds)
        self.scheduler.step()
        print('Epoch : {}, Train_loss : {}'.format(epo + 1, train_loss))
        self.bast_loss = save_checkpoint(self.model, self.bast_loss, train_loss, epo)










