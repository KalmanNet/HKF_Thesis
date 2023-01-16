# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________
from datetime import datetime as dt
import os

import matplotlib.pyplot as plt
import torch
import wandb
from torch import nn
from torch.utils.data.dataloader import DataLoader
from Logger.BaseLogger import WandbLogger,LocalLogger
from tqdm import trange
import numpy as np

class Pipeline():

    def __init__(self, modelName,Logger, gpu = True, **kwargs):



        Logs = {'Models':'.pt', 'Pipelines':'.pt','ONNX_models':'.onnx','CV_Loss':'.pdf', 'Train_Loss':'.pdf',
                'Sample_Plots':'.pdf'}

        if 'AdditionalLogs' in kwargs.keys(): Logs.update(kwargs['AdditionalLogs'])

        self.Logger = Logger

        self.Zoom = True

        if torch.cuda.is_available() and gpu:
            self.dev = torch.device("cuda:0")
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            print("using GPU!")
        else:
            self.dev = torch.device("cpu")
            print("using CPU!")

        self.Logger.AddLocalLogs(Logs)

        self.Base_folder  = os.path.dirname(os.path.realpath(__file__))
        self.modelName = modelName
        self.Time = dt.now()

        self.wandb = isinstance(self.Logger, WandbLogger)

        if 'hyperP' in kwargs.keys() :
            self.HyperParameters = kwargs['hyperP']
            assert isinstance(self.HyperParameters,dict), 'The Hyper-parameters must be given as a dict'


    # def UpdateHyperParameters(self, HyperP):
    #
    #     if hasattr(self,'HyperParameters'):
    #         self.HyperParameters.update(HyperP)
    #     else:
    #         self.HyperParameters = HyperP
    #
    #     if self.wandb:
    #         wandb.config.update(self.HyperParameters)

    def save(self):
        torch.save(self, self.Logger.GetLocalSaveName('Pipelines'))

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):

        self.model = model

        if hasattr(model,'ssModel'):
            self.setssModel(model.ssModel)

        if self.wandb: wandb.watch(self.model, log_freq = 10)



    def setTrainingParams(self, n_Epochs = 100, n_Batch = 32, learningRate = 1e-3,
                          weightDecay = 1e-6, shuffle = True, split_ratio = 0.7, loss_fn = nn.MSELoss(reduction='mean')):

        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch # Number of Samples in Batch
        self.learningRate = learningRate # Learning Rate
        self.weightDecay = weightDecay # L2 Weight Regularization - Weight Decay
        self.shuffle = shuffle # If we want to shuffle the data for each batch
        self.Train_CV_split_ratio = split_ratio

        # MSE LOSS Function
        self.loss_fn = loss_fn

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

        HyperP = {'lr': self.learningRate,
                  'BatchSize': self.N_B,
                  'Shuffle': str(self.shuffle),
                  'Train\CV Split Ratio ': self.Train_CV_split_ratio,
                  'Loss Function': str(self.loss_fn),
                  'L2': self.weightDecay,
                  'Optimizer': 'ADAM'}

        self.Logger.SaveConfig(HyperP)




    def InitModel(self,Batch_size, **kwargs):
        raise NotImplementedError('Methode needs to be implemented outside of base-class')

    def Run_Inference(self,input,target,**kwargs):
        raise NotImplementedError('Methode needs to be implemented outside of base-class')

    def NNTrain(self, DataSet, epochs=None, **kwargs):

        start_time = dt.now()

        print(f'---Started training of {self.modelName}---')

        try:
            self._NNTrain(DataSet,epochs,**kwargs)

        except:
            self.Logger.ForceClose()
            raise

        end_time = dt.now()

        training_time = end_time-start_time

        print(f'---Training of {self.modelName} finished---')

        print(f'Training took: {training_time.seconds//3600} hours,'
              f' {(training_time.seconds // 60) % 60 } minutes, {training_time.seconds % 60} seconds')


    def _NNTrain(self, DataSet, epochs=None, **kwargs):


        DataSet_length = len(DataSet)

        self.Logger.SaveConfig({'Training Set Size':DataSet_length})

        N_train = int(self.Train_CV_split_ratio * DataSet_length)
        N_CV = DataSet_length - N_train

        self.Logger.SaveConfig({'Train Samples':N_train,
                                'CV Samples':N_CV})


        self.MSE_cv_linear_epoch = torch.empty([self.N_Epochs], requires_grad=False).to(self.dev, non_blocking=True)
        self.MSE_cv_dB_epoch = torch.empty([self.N_Epochs], requires_grad=False).to(self.dev, non_blocking=True)

        self.MSE_train_linear_epoch = torch.empty([self.N_Epochs], requires_grad=False).to(self.dev, non_blocking=True)
        self.MSE_train_dB_epoch = torch.empty([self.N_Epochs], requires_grad=False).to(self.dev, non_blocking=True)

        self.MSE_batches = []

        ##############
        ### Epochs ###
        ##############

        MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        if epochs is None:
            N = self.N_Epochs
        else:
            N = epochs

        self.Logger.SaveConfig({'Epochs': N})

        sample_input, sample_target = DataSet[np.random.randint(0, len(DataSet))]

        Epoch_itter = trange(N)

        torch.random.manual_seed(420)

        Train_Dataset, CV_Dataset = torch.utils.data.random_split(DataSet, [N_train, N_CV],
                                                                  generator=torch.Generator(device=self.dev))

        self.InitTraining(N_CV)

        for ti in Epoch_itter:
            # torch.autograd.set_detect_anomaly(True)

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()

            CV_size = min(N_CV, 128)

            CV_Dataloader = DataLoader(CV_Dataset, shuffle=False, batch_size=CV_size)


            self.InitModel(N_CV,**kwargs)

            batch_cv_loss = 0.

            for cv_input, cv_target in CV_Dataloader:

                Inference_out, cv_loss = self.Run_Inference(cv_input, cv_target, **kwargs)

                batch_cv_loss += cv_loss


            self.MSE_train_linear_epoch[ti] = batch_cv_loss.detach()
            self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(batch_cv_loss).detach()

            Epoch_cv_loss_lin = batch_cv_loss.item()

            if (self.MSE_cv_dB_epoch[ti] < MSE_cv_dB_opt):

                MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                self.MSE_cv_idx_opt = ti

                torch.save(self.model, self.Logger.GetLocalSaveName('Models'))

                # torch.onnx.export(self.model,cv_input,self.Logger.GetLocalSaveName('ONNX_models'))


            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Training Mode
            self.model.train()

            Train_DataLoader = DataLoader(Train_Dataset, batch_size=self.N_B, shuffle=self.shuffle,
                                          generator=torch.Generator(device=self.dev))

            torch.random.manual_seed(42)


            # MSE_train_linear_batch = torch.empty(Train_DataLoader.__len__(), device=self.dev, requires_grad=False)
            MSE_batch_current_Epochs  = []

            self.InitTraining(self.N_B)

            for j, (train_input, train_target) in enumerate(Train_DataLoader):

                if not isinstance(train_input,list) and not train_input.shape[0] == self.N_B:
                    continue

                self.InitModel(self.N_B, **kwargs)

                Inference_out, train_loss = self.Run_Inference(train_input, train_target, **kwargs)

                self.MSE_batches.append(train_loss.detach().cpu().item())
                MSE_batch_current_Epochs.append(train_loss.detach().cpu().item())

                self.optimizer.zero_grad()

                train_loss.backward(retain_graph=False)

                self.optimizer.step()

            # Average
            self.MSE_train_linear_epoch[ti] = np.mean(MSE_batch_current_Epochs)#MSE_train_linear_batch.mean().detach()
            self.MSE_train_dB_epoch[ti] = 10*np.log10(self.MSE_train_linear_epoch[ti].cpu().detach())#10 * torch.log10(MSE_train_linear_batch.mean()).detach()

            Epoch_train_loss_lin = self.MSE_train_linear_epoch[ti].item()

            Epoch_cv_loss_dB = 10 * np.log10(Epoch_cv_loss_lin)
            Epoch_train_loss_dB = 10 * np.log10(Epoch_train_loss_lin)

            if self.wandb:

                wandb.log({'Training Loss': Epoch_train_loss_lin,'CV Loss': Epoch_cv_loss_lin,
                           'Training Loss [dB]': Epoch_train_loss_dB,'CV Loss [dB]': Epoch_cv_loss_dB
                           })


            # Update Description
            train_desc = str(round(Epoch_train_loss_dB, 4))
            cv_desc = str(round(Epoch_cv_loss_dB, 4))
            cv_best_desc = str(round(self.MSE_cv_dB_epoch[self.MSE_cv_idx_opt].item(),4))
            Epoch_itter.set_description(
                'Epoch training Loss: {} [dB], Epoch Val. Loss: {} [dB], Best Val. Loss: {} [dB]'.format(train_desc,
                                                                                                         cv_desc,
                                                                                                         cv_best_desc))
            if ti % 35 == 0 and ti != 0:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

        # Store Model to wandb
        if self.wandb:
            wandb.save(self.Logger.GetLocalSaveName('ONNX_models'),policy = 'now')
            wandb.save(self.Logger.GetLocalSaveName('Models'), policy = 'now')


        # Plot CV Loss
        plt.plot(self.MSE_cv_dB_epoch.detach().cpu().numpy(),'*', label = 'CV Loss', color = 'r' )
        plt.xlabel('Iteration')
        plt.ylabel('MSE Loss [dB]')
        plt.legend()
        plt.grid()
        plt.savefig(self.Logger.GetLocalSaveName('CV_Loss'))

        if not self.wandb:
            plt.show()
        plt.clf()

        # Plot Train Loss
        train_losses = np.array(self.MSE_batches)
        not_outliers = np.abs(train_losses-train_losses.mean()) < 2*train_losses.std()



        plt.plot(10*np.log10(train_losses[not_outliers]), '*', label='Train Batch Loss', color='r')
        plt.xlabel('Batch')
        plt.ylabel('MSE Loss [dB]')
        plt.legend()
        plt.grid()
        plt.savefig(self.Logger.GetLocalSaveName('Train_Loss'))

        if not self.wandb:
            plt.show()
        plt.clf()

        # Save Pipeline
        self.save()

        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]


    def NNTest(self, DataSet, **kwargs):


        start_time = dt.now()

        print(f'---Started testing of {self.modelName}---')

        try:
            self._NNTest(DataSet,**kwargs)

            if self.wandb:
                # self.Logger.SaveConfig(self.HyperParameters)


                intermediate = 10*torch.log10(self.MSE_test_linear_arr).detach().cpu().numpy()

                table = wandb.Table(data = [[x] for x in intermediate],columns = ['MSE Loss [dB]'])
                wandb.log({'Linear Test Loss Distribution': wandb.plot.histogram(table,'MSE Loss [dB]', title =
                                                                          'Test Loss Histogram')})

        except:
            self.Logger.ForceClose()
            raise

        end_time = dt.now()

        training_time = end_time-start_time

        print(f'---Testing of {self.modelName} finished---')

        print(f'Testing took: {training_time.seconds//3600} hours,'
              f' {(training_time.seconds // 60) % 60 } minutes, {training_time.seconds % 60} seconds')


    def _NNTest(self,Test_Dataset, **kwargs):

        N_T = len(Test_Dataset)

        self.Logger.SaveConfig({'Test Set Size':N_T})


        self.MSE_test_linear_arr = torch.empty([N_T],requires_grad= False)

        if hasattr(self.loss_fn,'reduction'):
            self.loss_fn.reduction = 'none'


        Model = torch.load(self.Logger.GetLocalSaveName('Models'),map_location= self.dev)

        Model.eval()

        Test_Dataloader = DataLoader(Test_Dataset, shuffle=False, batch_size= N_T)

        self.Overlaps = Test_Dataset.dataset.Overlap

        with torch.no_grad():

            self.InitModel(N_T,**kwargs)


            for test_input, test_target in Test_Dataloader:

                Inference_out, test_loss = self.Run_Inference(test_input, test_target, **kwargs)

                self.MSE_test_linear_arr = torch.mean(test_loss,dim=[n for n in range(1,test_loss.ndim)])

            num_plot_samples = 10

            observations,states = (Test_Dataset[-num_plot_samples:])

            observations = observations[:,0].cpu()
            states = states[:,0].cpu()

            results = [Inference_out[-num_plot_samples:,0].cpu()]
            labels = ['Autoencoder prediction']

            self.PlotResults(observations,states,results,labels,**kwargs)


            self.MSE_test_linear_avg = self.MSE_test_linear_arr.mean()
            self.MSE_test_dB_avg = 10*torch.log10(self.MSE_test_linear_avg)

            if self.wandb:
                wandb.log({'Test Loss [dB]': self.MSE_test_dB_avg.item()})


            print(f'Test Loss: {self.MSE_test_dB_avg} [dB]')

        self.save()

    def PlotResults(self, observations, states,results,labels,prefix,**kwargs):
        pass

    def InitTraining(self,Batch_size,**kwargs):
        pass