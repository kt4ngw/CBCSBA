


class Cost(object):
    def __init__(self):
        self.accumulated_energy = 0
        self.accumulated_latency = 0
    

    def get_latency_sum(self, selected_clients, bandwidth_allocation_result, round_i):
        latency_sum = 0
        latency_local_sum = 0
        latency_upload_sum = 0
        local = [0 for _ in range(len(selected_clients))]
        upload = [0 for _ in range(len(selected_clients))]
        for i in range(len(selected_clients)):
            local[i] = selected_clients[i].getLocalDelay(round_i)
            upload[i] = selected_clients[i].getUploadDelay(round_i, bandwidth_allocation_result[i])
            if local[i] + upload[i] > latency_sum:
                latency_sum = local[i] + upload[i]
            if local[i] > latency_local_sum:
                latency_local_sum = local[i]
            if upload[i] > latency_upload_sum:
                latency_upload_sum = upload[i]
        return (latency_sum, latency_local_sum, latency_upload_sum)

    def get_latency_sum_fccsba(self, selected_clients, bandwidth_allocation_result, round_i):
        latency_sum = 1000
        min_index = None
        local = [0 for _ in range(len(selected_clients))]
        upload = [0 for _ in range(len(selected_clients))]
        for i in range(len(selected_clients)):
            local[i] = selected_clients[i].getLocalDelay(round_i)
            upload[i] = selected_clients[i].getUploadDelay(round_i, bandwidth_allocation_result[i])
            if local[i] + upload[i] < latency_sum:
                latency_sum = local[i] + upload[i]
                min_index = i 
        return latency_sum, min_index 

        
    def get_energy_sum(self, selected_clients, bandwidth_allocation_result, round_i):
        energy_sum = 0
        energy_local_sum = 0
        energy_upload_sum = 0
        local = [0 for _ in range(len(selected_clients))]
        upload = [0 for _ in range(len(selected_clients))]
        for i in range(len(selected_clients)):
            local[i] = selected_clients[i].getLocalEngery(round_i)
            upload[i] = selected_clients[i].getUploadEngery(round_i, bandwidth_allocation_result[i])
            energy_sum += (local[i] + upload[i])
            energy_local_sum += local[i]
            energy_upload_sum += upload[i]
        return (energy_sum, energy_local_sum, energy_upload_sum)
            



class ClientAttr(object):
    def __init__(self, cpu_frequency, B, transmit_power):
        self.cpu_frequency = cpu_frequency
        self.bandwidth = B
        self.transmit_power = transmit_power

    def get_client_attr(self, id):
        return {
            "cpu_frequency": self.cpu_frequency[id],
            "B": self.bandwidth,
            "transmit_power": self.transmit_power[id],
        }
