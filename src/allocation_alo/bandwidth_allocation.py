from src.utils.tool_utils import setup_seed

import numpy as np
setup_seed(2024)
class Bandwidth_Allocation():
    def __init__(self, options, total_bandwidth):
        self.options = options
        self.total_bandwidth = total_bandwidth
        self.latency_upper = 1000
        self.latency_lower = 0
        self.V = 0
    # def allocation_bandwidth(self, total_bandwidth, selected_clients, V):
    #     result = [0 for i in range(len(selected_clients))]
    #     latency_A = self.latency_upper
    #     while(V = 0):
    #         for i in range(len(selected_clients)):
    #             result[i] = 0 # [7]中的公式（8）计算得出。
    #         allocated_bandwidth = sum(result)
    #         if allocated_bandwidth >  self.total_bandwidth:
    #             latency_A = (latency_A + self.latency_upper) / 2
    #             self.latency_lower = latency_A
    #         else if allocated_bandwidth < alpha * B:
    #             V = 1
    #             spare_bandwidth = total_bandwidth - allocated_bandwidth
    #             # ---- # 补充其他分配
    #             new_result = self.energy_bandwidth_allocate(result, spare_bandwidth)
    #         else:
    #             latency_A = (latency_A + self.latency_lower) / 2
    #             self.latency_upper = latency_A
    #     return new_result 
    # def energy_bandwidth_allocate(self, result, spare_bandwidth, selected_clients):
        
    #     new_result = [0 for i in range(len(selected_clients))]
    #     for i in range(len(selected_clients)):
    #         new_result[i] = result[i] + () * spare_bandwidth

    #     return new_result 


    def equal_allocation(self, selected_clients):
        bandwitdh_allocation_result = [1 / len(selected_clients) \
                                        * self.total_bandwidth for i in range(len(selected_clients))]

        return bandwitdh_allocation_result


    def proposed_bandwidth_allocation(self, selected_clients, round_i, baseline2021=False):
        latency_upper = 1000
        latency_lower = max([selected_clients[i].getLocalDelay(round_i) for i in range(len(selected_clients))])
        result = [0 for i in range(len(selected_clients))]
        latency_A = latency_upper
        V = 0 
        while(V == 0):
            for i in range(len(selected_clients)):
                result[i] = self.proposed_ba_comp_allcaotion(selected_clients[i], latency_A, round_i)
            allocated_bandwidth = sum(result)
            if baseline2021 == True:
               self.options["weight"] = 1 
            if allocated_bandwidth <  ((self.options["weight"] * (1 - 0.6)  + 0.6) * self.total_bandwidth) and allocated_bandwidth > ((self.options["weight"] * (1 - 0.6)  + 0.6) - 0.01) * self.total_bandwidth:
                V = 1
            else:
                if allocated_bandwidth > (self.options["weight"] * (1 - 0.6)  + 0.6) * self.total_bandwidth:
                    latency_lower = latency_A
                    latency_A = (latency_A + latency_upper) / 2

                else:
                    latency_upper = latency_A   
                    latency_A = (latency_A + latency_lower) / 2
        spare_bandwidth = self.total_bandwidth - allocated_bandwidth 
        new_result = self.energy_bandwidth_allocate(result, spare_bandwidth, selected_clients)    
        return new_result   

    def energy_bandwidth_allocate(self, result, spare_bandwidth, selected_clients):
        # allocation = [result[i] / sum(result) for i in range(len(result))]
        temp = 0
        for i in range(len(selected_clients)):
            # t = (result[i] * np.log2(1 + selected_clients[i].attr_dict['transmit_power'] * 8)) ** 2
            # temp += 1 / t
            t = (self.options['model_size'] * selected_clients[i].attr_dict['transmit_power']) / (result[i] * self.total_bandwidth)  ** 2 * np.log2(1 + selected_clients[i].attr_dict['transmit_power'] * 8)
            temp += t
        allocation = [((self.options['model_size'] * selected_clients[i].attr_dict['transmit_power']) / (result[i] * self.total_bandwidth)  ** 2 * np.log2(1 + selected_clients[i].attr_dict['transmit_power'] * 8)) / temp for i in range(len(result))]
        #print("allocation", allocation)
        # 
        #print("result", result)               
        new_result = [0 for i in range(len(selected_clients))]
        for i in range(len(selected_clients)):
          #  print("spare_bandwidth", spare_bandwidth)
            #print((allocation[i]) * spare_bandwidth)
            new_result[i] = result[i] + (allocation[i]) * spare_bandwidth
        return new_result 

    def proposed_ba_comp_allcaotion(self, client, latency_A, round_i):
        need_bandwidth = self.options['model_size'] / (((np.log2(1 + client.attr_dict['transmit_power'] * 8) * 1000000) / (8 * 1024 * 1024)) * (latency_A - client.getLocalDelay(round_i))) 
        # need_allocation_bandwidth = need_bandwidth / self.total_bandwidth
        return need_bandwidth
        # get 其计算时间


    def baseline2021_bandwidth_allocation(self, selected_clients, round_i, baseline2021=True):

        latency_upper = 2000
        latency_lower = max([selected_clients[i].getLocalDelay(round_i) for i in range(len(selected_clients))])
        result = [0 for i in range(len(selected_clients))]
        latency_A = latency_upper
        V = 0 
        while(V == 0):
            for i in range(len(selected_clients)):
                result[i] = self.proposed_ba_comp_allcaotion(selected_clients[i], latency_A, round_i)
            allocated_bandwidth = sum(result)
            # if baseline2021 == True:
            #    self.options["weight"] = 1 
            if allocated_bandwidth <  (1 * self.total_bandwidth) and allocated_bandwidth > (1- 0.01) * self.total_bandwidth:
                V = 1
            else:
                if allocated_bandwidth > 1 * self.total_bandwidth:
                    latency_lower = latency_A
                    latency_A = (latency_A + latency_upper) / 2

                else:
                    latency_upper = latency_A   
                    latency_A = (latency_A + latency_lower) / 2 

        return result  

    def jcsba(self, selected_clients, selected_index_latency):
        print(selected_index_latency)
        result = [selected_index_latency[i] / sum(selected_index_latency) * self.total_bandwidth for i in range(len(selected_clients))]
        print(result)
        return result


    def jacsba_one_allocation(self, selected_clients):
        result = [1.0 * self.total_bandwidth]
        return result