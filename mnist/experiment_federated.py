from environment_federated import *
import matplotlib.pyplot as plt



def normal_training_exp(sim_num, peer_pseudonym,
                peer_num, peer_frac, seed, tau,
               global_rounds, local_epochs, local_bs,
               local_lr , local_momentum , num_classes , 
               labels_dict, classes_list, device):

               print('\n===> Start Normal Training Simulation in Simple Environment...')
               
               simple = SimpleEnv(num_peers=peer_num, peer_pseudonyms = peer_pseudonym, peer_frac = peer_frac, seed = seed, tau = tau,
               global_rounds = global_rounds, local_epochs = local_epochs, local_bs = local_bs, local_lr = local_lr, local_momentum = local_momentum,
               num_classes = num_classes, labels_dict = labels_dict, classes_list = classes_list, device = device)
               
               for i in range(sim_num):
                   simple.simulate(i+1)
                #    print('\nmarks in this simulation')
                #    print(simple.simple_marks)

def untargeted_attack_training_exp(sim_num, peer_pseudonym,
                peer_num, peer_frac , seed,tau,
               global_rounds, local_epochs, local_bs,
               local_lr , local_momentum , num_classes , 
               labels_dict, classes_list, device, attack_type, attack_rates, malicious_behaviour_rate):

            print('\n==>Start untargeted attack Simulation in Simple Environment...\n')

            for attack_rate in attack_rates:

                print('\n===>Untargeted attack with rate of: ({:.0f}%) of peers and malicious behaviour rate of: ({:.0f}%)'.format(attack_rate*100, malicious_behaviour_rate*100))

                simple = SimpleEnv(num_peers=peer_num, peer_pseudonyms = peer_pseudonym, peer_frac = peer_frac, seed = seed, tau = tau, 
                global_rounds = global_rounds, local_epochs = local_epochs, local_bs = local_bs, local_lr = local_lr, local_momentum = local_momentum,
                num_classes = num_classes, labels_dict = labels_dict, classes_list = classes_list, device = device, 
                attack_type = attack_type, attack_rate = attack_rate, malicious_behaviour_rate = malicious_behaviour_rate)
                
                for i in range(sim_num):
                    simple.simulate(i+1)

def targeted_attack_training_exp(sim_num, peer_pseudonym,
                peer_num, peer_frac , seed,tau,
               global_rounds, local_epochs, local_bs,
               local_lr , local_momentum , num_classes , 
               labels_dict, classes_list, device, attack_type, attack_rates, malicious_behaviour_rate, mapping_list,
               maximum_attacks):

            print('\n==>Start targeted attack Simulation in Simple Environment...\n')

            for attack_rate in attack_rates:

                print('\n===>Targeted attack with rate of: ({:.0f}%) of peers and malicious behaviour rate of: ({:.0f}%)'.format(attack_rate*100, malicious_behaviour_rate*100))

                simple = SimpleEnv(num_peers=peer_num, peer_pseudonyms = peer_pseudonym, peer_frac = peer_frac, seed = seed, tau = tau, 
                global_rounds = global_rounds, local_epochs = local_epochs, local_bs = local_bs, local_lr = local_lr, local_momentum = local_momentum,
                num_classes = num_classes, labels_dict = labels_dict, classes_list = classes_list, device = device, 
                attack_type = attack_type, attack_rate = attack_rate, malicious_behaviour_rate = malicious_behaviour_rate , mapping_list = mapping_list,
                maximum_attacks = maximum_attacks)
                
                for i in range(sim_num):
                    simple.simulate(i+1)


#Training in Secure Environment

def secureEnv_random_attack_training_exp(sim_num, peer_pseudonym,
                peer_num, peer_frac, seed, tau, 
               global_rounds, local_epochs, local_bs,
               local_lr , local_momentum , num_classes , 
               labels_dict, classes_list, device,
                attack_type, attack_rates, mapping_list, malicious_behaviour_rate):

            print('\n==>Start Untargeted attack Simulation in EigentTrust Environment...\n')

            for attack_rate in attack_rates:

                print('\n===>Untargeted attack with rate of: ({:.0f}%) of peers and malicious behaviour rate of: ({:.0f}%)'.format(attack_rate*100, malicious_behaviour_rate*100))

                secure_env = SecureEnv(num_peers=peer_num, peer_pseudonyms = peer_pseudonym, peer_frac = peer_frac, seed = seed, tau = tau, 
                 global_rounds = global_rounds, local_epochs = local_epochs, local_bs = local_bs, local_lr = local_lr, local_momentum = local_momentum,
                num_classes = num_classes, labels_dict = labels_dict, classes_list = classes_list, device = device, 
                attack_type = attack_type, attack_rate = attack_rate, malicious_behaviour_rate = malicious_behaviour_rate, mapping_list = mapping_list)
                
                for i in range(sim_num):
                    secure_env.simulate(i+1)


def secureEnv_targeted_attack_training_exp(sim_num, peer_pseudonym,
                peer_num, peer_frac, seed, tau, 
               global_rounds, local_epochs, local_bs,
               local_lr , local_momentum , num_classes , 
               labels_dict, classes_list, device,
                attack_type, attack_rates, mapping_list, malicious_behaviour_rate,
                 maximum_attacks):

            print('\n==>Start targeted attack Simulation in Secure Environment...\n')

            for attack_rate in attack_rates:

                print('\n===>targeted attack with rate of: ({:.0f}%) of peers and malicious behaviour rate of: ({:.0f}%)'.format(attack_rate*100, malicious_behaviour_rate*100))

                secure_env = SecureEnv(num_peers=peer_num, peer_pseudonyms = peer_pseudonym, peer_frac = peer_frac, seed = seed, tau = tau, 
                 global_rounds = global_rounds, local_epochs = local_epochs, local_bs = local_bs, local_lr = local_lr, local_momentum = local_momentum,
                num_classes = num_classes, labels_dict = labels_dict, classes_list = classes_list, device = device, 
                attack_type = attack_type, attack_rate = attack_rate, mapping_list = mapping_list, malicious_behaviour_rate = malicious_behaviour_rate,
                maximum_attacks = maximum_attacks)
                
                for i in range(sim_num):
                    secure_env.simulate(i+1)


    
    
   


