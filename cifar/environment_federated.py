from __future__ import print_function
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from dataset import Dataset
from models import CNNMnist, CNNCifar
import os
from torch.utils import data
import random
#from update import test_inference
from tqdm import tqdm
from tqdm import tqdm_notebook
import statistics
import copy
from utils import *
from vgg_models import *
from operator import itemgetter
import time

NUM_CLASSES = 10
NUM_PEERS = 100


class Peer():
    

    def __init__(self, peer_id, peer_pseudonym, device, local_epochs, local_bs, local_lr, local_momentum, num_peers,tau = 1.5, 
    peer_type = 'Honest', mapping_list = None, malicious_behaviour_rate = 0, noise_mean = None, noise_sd = None):
        
        
        _performed_attacks = 0
        self.peer_id = peer_id
        self.peer_pseudonym = peer_pseudonym
        self.device = device
        self.local_epochs = local_epochs
        self.local_bs = local_bs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.num_peers = num_peers
        self.tau = tau
        self.peer_type = peer_type
        self.malicious_behaviour_rate = malicious_behaviour_rate
        self.mapping_list = mapping_list
        self.noise_mean = noise_mean
        self.noise_sd = noise_sd
        
       

        
          
#======================================= Start of training function ===========================================================#
    # Tasks numbers (0: Normal training, 1: untargeted poisoning trainingt,  2: stealthy poisononing training)
    def train_and_update(self, model, dataset_fraction, global_epoch, maximum_attacks) :

             
        train_loader = torch.utils.data.DataLoader(
            dataset_fraction,
            batch_size=self.local_bs,
            shuffle=True,
            num_workers=1)

        optimizer = optim.SGD(model.parameters(), lr=self.local_lr, momentum = self.local_momentum)

        
        model.train()
        epoch_loss = []

        print('\n{} starts training in Global Round:{} |'.format((self.peer_pseudonym), (global_epoch + 1)))
        
        f = 0
        if(self.peer_type == 'Stealthy_Attacker' and type(self).performed_attacks<maximum_attacks):
            print('Performed attacks', self.performed_attacks)
            r = np.random.rand()
            if(r <= self.malicious_behaviour_rate):
                f = 1
                print('\n===>Targeted/stealthy attack started by: ', self.peer_pseudonym, ' in Global Round: ', global_epoch + 1)

        for epochs in tqdm_notebook(range(self.local_epochs)):
            batch_loss = []
            for batch_idx, (data, target) in enumerate(train_loader):
                if(f == 1):
                    for i in range(len(target)):
                        target[i] = self.mapping_list[target[i]]
                #data = data.transpose(3, 1)
               
                data=data.float()
                data, target = data.to(self.device), target.to(self.device)
                #poisoning attack example
                # if(self.worker_id == 0 and target[0] == 0):
                #     target = 9
                model.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()            
                optimizer.step()
                batch_loss.append(loss.item())
                if (batch_idx == 0 or batch_idx == len(train_loader)-1):
                    print('Train Epoch: {} [{}/{}\t({:.0f}%)]\tLoss: {:.6f}'.format(
                    (epochs+1), batch_idx * len(data), len(train_loader.dataset),
                    100. * (batch_idx / len(train_loader))+1, loss.item()))

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        if(self.peer_type == 'Random_Attacker' and type(self).performed_attacks<maximum_attacks):
            print('Performed attacks', self.performed_attacks)
            r = np.random.rand()
            if(r <= self.malicious_behaviour_rate):
                print('\n===>Random attack started by: ', self.peer_pseudonym, ' in Global Round: ', global_epoch+1)
                w = model.state_dict()
                print("Random attack strating: ")
                
                for key in w.keys():
                    if(len(w[key].size()) != 0):
                        noise = (torch.tensor(np.random.normal(self.noise_mean, self.noise_sd, size=w[key].size(),))).to(self.device)
                        w[key]+= noise
                        del noise
                model.load_state_dict(w)
                type(self).performed_attacks+=1
                print('Performed attacks', self.performed_attacks)
        
        if f == 1:
                type(self).performed_attacks+=1
                print('Performed attacks', self.performed_attacks)

        print('{} ends training in Global Round:{} |'.format((self.peer_pseudonym), (global_epoch+1)))

        return copy.deepcopy(model.state_dict()), sum(epoch_loss) / len(epoch_loss), copy.deepcopy(model)
#======================================= End of training function =============================================================#





#======================================= Start of marking function =============================================================#

    def get_score(self, normalized_distances, distance):
       
        q1, q3, iqr = get_quartiles(sorted(normalized_distances))
        #print('Q1: {}\t Median/Q2: {}\t Q3: {}\t IQR: {}\t'.format(Q1, Median, Q3, IQR))
        lower_bound = q1 -(self.tau * iqr) 
        upper_bound = q3 + (self.tau * iqr)

        if(distance <= upper_bound):
            return 1    
        else:
            return 0
#======================================= End of marking function =============================================================#



#======================================= Start of requesting forwarding updates function =====================================#
    def forwarding_request(task_id, global_round):
        return 0

#======================================= End of requesting forwarding updates function =======================================#



class SimpleEnv:

    def __init__(self, num_peers, peer_pseudonyms, peer_frac, seed, tau, global_rounds, local_epochs, local_bs, local_lr,
    local_momentum, num_classes, labels_dict, classes_list, device, 
    attack_type = None, attack_rate = None, mapping_list = None, malicious_behaviour_rate = None,
    noise_mean = None, noise_sd = None, maximum_attacks = None):
        

        SimpleEnv._credits = np.ones(num_peers)
        self.num_peers = num_peers
        self.peer_pseudonyms = peer_pseudonyms
        self.peer_frac = peer_frac
        self.seed = seed
        self.tau = tau
        self.global_rounds = global_rounds
        self.local_epochs = local_epochs
        self.local_bs = local_bs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.num_classes = num_classes
        self.labels_dict = labels_dict
        self.classes_list = classes_list
        self.device = device
        self.attack_type = attack_type
        self.attack_rate = attack_rate
        self.malicious_behaviour_rate = malicious_behaviour_rate
        self.noise_mean = noise_mean
        self.noise_sd = noise_sd
        self.maximum_attacks = maximum_attacks
        self.mapping_list = mapping_list

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
       

        #Change the directory to the place of the application
        os.chdir("D:\Federated_Learning\Applications\Experiments of March2020\Federated_Learning_EigentTrust_100_workers_GPU\cifar/")
        self.PATH = os.path.join(os.getcwd(), 'Checkpoints')

        #create the global model
        print('======>Creating Global model.....')
        self.global_model = VGG('VGG16')
        # self.global_model = CNNCifar(10)
        

        #if the global model is not found, save its state, else load the saved model
        if not (os.path.exists(os.path.join(self.PATH, 'CIFAR_global_model.pth'))):
            torch.save(self.global_model.state_dict(), os.path.join(self.PATH, 'CIFAR_global_model.pth'))
        else:
            self.global_model.load_state_dict(torch.load(os.path.join(self.PATH, 'CIFAR_global_model.pth')))
        
        #global_model = DenseNet(784, 50, 10)
        self.global_model.to(self.device)
        self.global_model.train()
        print(self.global_model)
        
        self.peers_list = []
       
        random_attackers, stealthy_attackers = [], []
        if(self.attack_type == 'untargeted'):
            #pick m random workers from the workers list
            n = max(int(self.attack_rate * self.num_peers), 1)
            random_attackers = np.random.choice(range(self.num_peers), n, replace=False)
        
        if(self.attack_type == 'targeted'):
            #pick m random workers from the workers list
            n = max(int(self.attack_rate * self.num_peers), 1)
            stealthy_attackers = np.random.choice(range(self.num_peers), n, replace=False)
        

        for i in range(self.num_peers):
            if i in random_attackers:
                
                self.peers_list.append(Peer(i, self.peer_pseudonyms[i], self.device, self.local_epochs, self.local_bs, self.local_lr, self.local_momentum,
                self.num_peers, tau = self.tau, peer_type='Random_Attacker', malicious_behaviour_rate= self.malicious_behaviour_rate,
                noise_mean = self.noise_mean, noise_sd = self.noise_sd))

            elif i in stealthy_attackers:
                self.peers_list.append(Peer(i, self.peer_pseudonyms[i], self.device, self.local_epochs, self.local_bs, self.local_lr, self.local_momentum,
               self.num_peers, tau = self.tau, peer_type='Stealthy_Attacker', malicious_behaviour_rate= self.malicious_behaviour_rate, mapping_list=self.mapping_list))

            else:
                self.peers_list.append(Peer(i, self.peer_pseudonyms[i], self.device, self.local_epochs, self.local_bs, self.local_lr, self.local_momentum,
                num_peers = self.num_peers, tau = self.tau))


    
    #=============================== Start of fixing the environment seed to reproduce a deterministic results ========================#

        

        self.training_dataset_list = []
        self.testing_dataset_list = []
        self.global_train_dataset = None
        self.global_test_dataset = None
        self.test_loader = None

    
    @property
    def credits(self):
        return self.__class__._credits
    @credits.setter
    def credits(self, value):
        self.__class__._credits = value

    #print model trainable parameters
    def print_env_model_learnable_parameters(self):

        total = 0
        print('\nTrainable parameters:')
        for name, param in global_model.named_parameters():
            if param.requires_grad:
                print(name, '\t', param.numel())
                total += param.numel()
        print()
        print('Total', '\t', total, "trainable parametsers")


    def split_dataset(self):
        #======================================= Start of distribution and loading of the datasets ====================================#
        print("=======> loading training dataset......\n")
        #paths to training and testing datasets
        train_path = str(os.getcwd()+"/datasets/training/")
        test_path = str(os.getcwd() + "/datasets/testing")

        # Define a transforms to normalize and augment the data
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        

        self.global_train_dataset = Dataset(train_path, transform_train, 0, 50000, 32, False)
        self.global_test_dataset = Dataset(test_path, transform_test,  0, 10000, 32, False) 


        for i in range(self.num_peers):
            self.training_dataset_list.append(Dataset(train_path, transform_train, i*500, (i+1)*500, 32, True))

        self.test_loader = data.DataLoader(
            self.global_test_dataset,
            batch_size=100,
            shuffle=False,
            num_workers=1)

        print("\nLoading done!....\n")
        # for i in range(self.num_peers):
        #    print(len(self.training_dataset_list[i]))
#======================================= End of distribution and loading of the datasets =======================================#


#======================================= Start of testning function ===========================================================#
    def test(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:              
                data=data.float()
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return (float(correct) / len(test_loader.dataset))
    #======================================= End of testning function =============================================================#


#Test label prediction function    
    def test_label_predictions(self, model, device, test_loader):
        model.eval()
        actuals = []
        predictions = []
        with torch.no_grad():
            for data, target in test_loader:
                
    #             img = data[0]
    #             img = transforms.ToPILImage()(img)
    #             plt.imshow(img, cmap='gray')
    #             plt.show()
    #             print(target[0])
    #             break

                data=data.float()
                data, target = data.to(device), target.to(device)
                output = model(data)
                prediction = output.argmax(dim=1, keepdim=True)
                actuals.extend(target.view_as(prediction))
                predictions.extend(prediction)
        return [i.item() for i in actuals], [i.item() for i in predictions]

    #plot simulation step losses
    def plot_losses(self, losses):

        plt.figure(figsize=(14, 4))
        plt.xlabel('Training epoch')
        plt.ylabel('Loss')
        plt.plot([loss for loss in losses])
        plt.show()
    
    def plot_accuracies(self, accuracies):
        plt.figure(figsize=(15, 4))
        plt.xticks(range(len(accuracies)))
        plt.xlabel('training epoch')
        plt.ylabel('testing accuracy')
        plt.plot(accuracies, marker='o')
        plt.show()

    
    def test_class_probabilities(self, model, device, test_loader, which_class):
        model.eval()
        actuals = []
        probabilities = []
        with torch.no_grad():
            for data, target in test_loader:
                data=data.float()
                data, target = data.to(device), target.to(device)
                output = model(data)
                output = output.data.cpu()
                prediction = output.argmax(dim=1, keepdim=True)
                actuals.extend(target.view_as(prediction) == which_class)
                probabilities.extend(np.exp(output[:, which_class]))
        return [i.item() for i in actuals], [i.item() for i in probabilities]
    
    
    
    def plot_class_probabilities(self, which_class):

        actuals, class_probabilities = self.test_class_probabilities(self.global_model, self.device, self.test_loader, which_class)

        fpr, tpr, _ = roc_curve(actuals, class_probabilities)

        roc_auc = auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for label=Eight(%d) class' % which_class)
        plt.legend(loc="lower right")
        plt.show()


        #choose random requester and set of workers
    def choose_peers(self):
        
        #pick m random workers from the workers list
        m = max(int(self.peer_frac * self.num_peers), 1)
        selected_workers = np.random.choice(range(self.num_peers), m, replace=False)

        print('\nSelected workers\n')
        print(selected_workers+1)
        for w in selected_workers:
            print(self.peers_list[w].peer_pseudonym, ': is ', self.peers_list[w].peer_type)

        return selected_workers


            
    def simulate(self, sim_num):

        simulation_model = copy.deepcopy(self.global_model)
        print('\n===>Simulation Step:', sim_num)
        self.split_dataset()
       
        # copy weights
        global_weights = simulation_model.state_dict()

        targeted_class_accuracy = [[],[]]
        global_losses = []
        global_accuracies = []
        best_accuracy = 0.0

        #start training
        print("\n=======> Start Global Model Training......\n")
        
        for epoch in tqdm_notebook(range(self.global_rounds)):
            
            Peer.performed_attacks = 0

            
            selected_workers = self.choose_peers()

            local_weights,  local_losses, local_models = [], [], []
            print(f'\n | Global Training Round : {epoch+1} |\n')
            
            simulation_model.train()

            for selected_worker in selected_workers:             
                worker_weights, worker_loss, local_model = self.peers_list[selected_worker].train_and_update(copy.deepcopy(simulation_model),
                                                self.training_dataset_list[selected_worker],epoch, self.maximum_attacks)
                    
                
                local_weights.append(copy.deepcopy(worker_weights))
                local_losses.append(copy.deepcopy(worker_loss))   
                local_models.append(local_model)

            loss_avg = sum(local_losses) / len(local_losses)
            global_losses.append(loss_avg)

            
                            
            #aggregated global weights
            global_weights = average_weights(local_weights, [1 for i in range(len(selected_workers))])
            
            
            # update global weights
            simulation_model.load_state_dict(global_weights)
                    
            currenet_accuracy = self.test(simulation_model, self.device, self.test_loader)
            global_accuracies.append(currenet_accuracy)
            
            #print and show confusion matrix after each global round
            actuals, predictions = self.test_label_predictions(simulation_model, self.device, self.test_loader)
            print('Confusion matrix:')
            print(confusion_matrix(actuals, predictions))
            print('F1 score: %f' % f1_score(actuals, predictions, average='micro'))
            print('Accuracy score: %f' % accuracy_score(actuals, predictions))


            fig, ax = plt.subplots(1,1,figsize=(8,6))
            ax.matshow(confusion_matrix(actuals, predictions), aspect='auto', vmin=0, vmax=1000, cmap=plt.get_cmap('Blues'))
            plt.ylabel('Actual Category')
            plt.yticks(range(10), self.classes_list)
            plt.xlabel('Predicted Category')
            plt.xticks(range(10), self.classes_list)
            plt.show()


            print('{0:10s} - {1}'.format('Category','Accuracy'))
            for i, r in enumerate(confusion_matrix(actuals, predictions)):
                print('{0:10s} - {1:.1f}'.format(self.classes_list[i], r[i]/np.sum(r)*100))
                if i == 4:
                    targeted_class_accuracy[0].append(r[i]/np.sum(r)*100)
                if i == 7:
                    targeted_class_accuracy[1].append(r[i]/np.sum(r)*100)


            # # Save the new model just if there is an accuracy improvement
            # if currenet_accuracy > best_accuracy:
            #     best_accuracy = currenet_accuracy
            #     torch.save(simulation_model.state_dict(), os.path.join(self.PATH, 'global_model.pth'))          

        print('Average loss during gobal rounds')
        print(global_losses)
        self.plot_losses(global_losses)
        print('Accuracy during gobal rounds')
        print(global_accuracies)
        self.plot_accuracies(global_accuracies)
        print('Class deer accuracy')
        print(targeted_class_accuracy[0])
        self.plot_accuracies(targeted_class_accuracy[0])
        print('Class horse accuracy')
        print(targeted_class_accuracy[1])
        self.plot_accuracies(targeted_class_accuracy[1])
        #=================================End of model creation and distribute workers tasks ========================================#



class SecureEnv(SimpleEnv):
    
    def __init__(self, num_peers, peer_pseudonyms, peer_frac, seed, tau, global_rounds, local_epochs, local_bs, local_lr,
    local_momentum, num_classes, labels_dict, classes_list, device, 
    attack_type, attack_rate, mapping_list, malicious_behaviour_rate,noise_mean, noise_sd, maximum_attacks):
    
        
        SecureEnv._selected_before = []


        super().__init__(num_peers = num_peers, peer_pseudonyms = peer_pseudonyms, peer_frac = peer_frac, seed = seed, tau = tau, 
            global_rounds = global_rounds, local_epochs = local_epochs, local_bs = local_bs, local_lr = local_lr, local_momentum = local_momentum,
            num_classes = num_classes, labels_dict = labels_dict, classes_list = classes_list, device = device, 
            attack_type = attack_type, attack_rate = attack_rate, mapping_list = mapping_list, malicious_behaviour_rate = malicious_behaviour_rate,
            noise_mean = noise_mean, noise_sd = noise_sd, maximum_attacks = maximum_attacks)

        

    @property
    def selected_before(self):
        return self.__class__._selected_before
    @selected_before.setter
    def selected_before(self, value):
        self.__class__._selected_before = value

    
    def choose_peers(self, maximum_attacks):
        remaining_malicious = copy.deepcopy(maximum_attacks)
        
        #pick m random workers from the workers list
        m = max(int(self.peer_frac * self.num_peers), 1)
        selected_workers = np.random.choice(range(self.num_peers), m, replace=False)

        print('\nSelected workers\n')
        print(selected_workers+1)
        workers_classes = []
        for w in selected_workers:
            print(self.peers_list[w].peer_pseudonym, ': is ', self.peers_list[w].peer_type)
            if self.peers_list[w].peer_type != 'Honest' and remaining_malicious>0:
                workers_classes.append(0)
                remaining_malicious-=1
            else:
                workers_classes.append(1)

            

        return selected_workers, np.asarray(workers_classes)
                    

    
    def display_current_scores(self):
        
        for i in range(len(self.credits)):
            print(self.peers_list[i].peer_pseudonym, '  Credits: ', max(self.credits[i], 0))
    
    def simulate(self, sim_num):

        print('\n===>Simulation Step:', sim_num)
        print('Peers types in this Simulation |')

        # for i in range(self.num_peers):
        #     print(self.peers_list[i].peer_pseudonym, ' is: ', self.peers_list[i].peer_type)

        
        self.split_dataset()
        simulation_model = copy.deepcopy(self.global_model)
        # copy weights
        global_weights = simulation_model.state_dict()

        global_losses = []
        global_accuracies = []
        targeted_class_accuracy = [[],[]]
        detection_accuracy=[]
        best_accuracy = 0.0

        #start training
        print("\n=======> Start Global Model Training......\n")
        
        for epoch in tqdm_notebook(range(self.global_rounds)):
            
            Peer.performed_attacks = 0
            
            selected_workers, workers_classes = self.choose_peers(self.maximum_attacks)

            local_weights,  local_losses, local_models = [], [], []
           
            print(f'\n | Global Training Round : {epoch+1} |\n')
            
            
            simulation_model.train()


            for selected_worker in selected_workers:                
                worker_weights, worker_loss, local_model = self.peers_list[selected_worker].train_and_update(copy.deepcopy(simulation_model),
                                                self.training_dataset_list[selected_worker],epoch, self.maximum_attacks)
                    
                local_weights.append(copy.deepcopy(worker_weights))
                local_losses.append(copy.deepcopy(worker_loss))  
                local_models.append(local_model) 
            
            
            dic_worker_classified = {'1': 'Honest', '0': 'Malicious'}
            additional_time = 0
            start_time = time.time()



            #get global round centroid
            
            # centroid, distances, additional_time = get_current_round_centroid(local_weights)
            #get the normalized distances
            # normalized_distances = get_distances_to_centroid(local_weights, centroid, normalized=True)



            # #get_last_layer_biases
            last_layer_biases = get_layer_weights(local_weights, layer_name = 'classifier.bias')
            print('Last layer biases ', last_layer_biases)


            mad_biases = [compute_mad(last_layer_biases[i].flatten(), 10) for i in range(len(last_layer_biases))]
            print('MAD biases', mad_biases)
            geomed = geometric_median(last_layer_biases, method='auto')
            print('GeoMed ', geomed)
            distances_from_geomed = get_distances_from_geomed(geomed, last_layer_biases)[0]
            print('Distances from geomed ', distances_from_geomed)
            honest_workers, scores = get_honest_workers(distances_from_geomed)
            print(scores)
            np_scores = np.asarray(scores)
            detection_accuracy.append(((np_scores==workers_classes).sum())/len(scores))
            print('Detection accuracy {}%'.format(detection_accuracy[epoch]*100))
            
            
            #get a single Krum 
            # single_krum = krum(local_models, int (self.attack_rate*len(local_models)), False)
            # print('Krum best update ', single_krum)

             #get multi Krum 
            # multi_krum = krum(local_models, int (0.4*len(local_models)), True)
            # print('Krum best updates ', multi_krum)
            # best_updates = []
            # for x in multi_krum:
            #     best_updates.append(local_weights[x])

            # print('\nDistances from the centroid')
            # plot_box_whishker(normalized_distances)
            # # print('\nPlotting of biases')
            # plot_box_whishker(distances_from_geomed)
            # plot_box_whishker(mad_biases)

            i = 0
            # scores = []
            for selected_worker in selected_workers:
                print('\nWorker: ', self.peers_list[selected_workers[i]].peer_pseudonym)
                # score1 = self.peers_list[i].get_score(distances_from_geomed, distances_from_geomed[i])
                # #score2 = self.peers_list[i].get_score(mad_biases, mad_biases[i])
                # # self.credits[selected_worker] += score
                # # self.credits[selected_worker] = min(self.credits[selected_worker], 10)
                # score = score1#*score2
                # scores.append(max(score, 0))
                print('The worker is :', self.peers_list[selected_workers[i]].peer_type, ' and classified as: ', dic_worker_classified[str(scores[i])])
                print('Distance from GeoMed ', distances_from_geomed[i])
                i+=1
            
            loss_avg = sum(local_losses) / len(local_losses)
            
            global_losses.append(loss_avg)

           
            # #aggregated local weights with their accountability_managers_magority_ratings
            global_weights = average_weights(local_weights, scores)
            print("--- %s seconds ---" % ((time.time() - start_time)- additional_time))
                      
            # update global weights
            simulation_model.load_state_dict(global_weights)

            currenet_accuracy = self.test(simulation_model, self.device, self.test_loader)
            global_accuracies.append(currenet_accuracy)

            #Display workers marks after every global training epoch
            # print('Credits of Peers After simulation')
            # self.display_current_scores()
            
                   
            
            #print and show confusion matrix after each global round
            actuals, predictions = self.test_label_predictions(simulation_model, self.device, self.test_loader)
            print('Confusion matrix:')
            print(confusion_matrix(actuals, predictions))
            print('F1 score: %f' % f1_score(actuals, predictions, average='micro'))
            print('Accuracy score: %f' % accuracy_score(actuals, predictions))


            fig, ax = plt.subplots(1,1,figsize=(8,6))
            ax.matshow(confusion_matrix(actuals, predictions), aspect='auto', vmin=0, vmax=1000, cmap=plt.get_cmap('Blues'))
            plt.ylabel('Actual Category')
            plt.yticks(range(10), self.classes_list)
            plt.xlabel('Predicted Category')
            plt.xticks(range(10), self.classes_list)
            plt.show()


            print('{0:10s} - {1}'.format('Category','Accuracy'))
            for i, r in enumerate(confusion_matrix(actuals, predictions)):
                print('{0:10s} - {1:.1f}'.format(self.classes_list[i], r[i]/np.sum(r)*100))
                if i == 4:
                    targeted_class_accuracy[0].append(r[i]/np.sum(r)*100)
                if i == 7:
                    targeted_class_accuracy[1].append(r[i]/np.sum(r)*100)


            # # Save the new model just if there is an accuracy improvement
            # if currenet_accuracy > best_accuracy:
            #     best_accuracy = currenet_accuracy
            #     torch.save(simulation_model.state_dict(), os.path.join(self.PATH, 'global_model.pth'))          

        print('Average loss during gobal rounds')
        print(global_losses)
        self.plot_losses(global_losses)
        print('Accuracy during gobal rounds')
        print(global_accuracies)
        self.plot_accuracies(global_accuracies)
        print('Class deer accuracy')
        print(targeted_class_accuracy[0])
        self.plot_accuracies(targeted_class_accuracy[0])
        print('Class horse accuracy')
        print(targeted_class_accuracy[1])
        self.plot_accuracies(targeted_class_accuracy[1])
        print('Detection accuracy')
        print('Detection accuracy {}%'.format(sum(detection_accuracy)/len(detection_accuracy)*100))
        self.plot_accuracies(detection_accuracy)
        
    #=================================End of model creation and distribute workers tasks ========================================#
        
        

    






            

    












