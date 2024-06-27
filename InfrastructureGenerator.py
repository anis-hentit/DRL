import json
import random


'''
IoT devices are indexed from 0 to 29.
Edge servers are indexed from 30 to 44.
Cloud servers are indexed from 45 to 49.
'''

def generate_hosts():
    hosts_config = []
    # IoT Devices (0-29)
    for i in range(30):
        hosts_config.append({"CPU": random.randint(1, 5), "BW": random.randint(1, 3)})

    # Edge Servers (30-44)
    for i in range(15):
        hosts_config.append({"CPU": random.randint(40, 60), "BW": random.randint(20, 40)})

    # Cloud Servers (45-49)
    for i in range(5):
        hosts_config.append({"CPU": random.randint(200, 300), "BW": random.randint(80, 120)})

    return hosts_config

def generate_links():
    network_topology = []

    # IoT to IoT Links
    for i in range(30):
        for j in range(i + 1, 30):
            if random.random() < 0.2:  # 20% chance to create a link
                bandwidth = random.randint(30, 50)
                latency = random.randint(1, 10)
                network_topology.append({"source": i, "destination": j, "bandwidth": bandwidth, "latency": latency})
                network_topology.append({"source": j, "destination": i, "bandwidth": bandwidth, "latency": latency})

    # IoT to Edge Links
    for i in range(30):
        for j in range(30, 45):
            if random.random() < 0.9:  # 90% chance to create a link
                bandwidth = random.randint(50, 70)
                latency = random.randint(5, 15)
                network_topology.append({"source": i, "destination": j, "bandwidth": bandwidth, "latency": latency})
                network_topology.append({"source": j, "destination": i, "bandwidth": bandwidth, "latency": latency})

    # Edge to Edge Links
    for i in range(30, 45):
        for j in range(i + 1, 45):
            if random.random() < 0.6:  # 60% chance to create a link
                bandwidth = random.randint(80, 100)
                latency = random.randint(16, 25)
                network_topology.append({"source": i, "destination": j, "bandwidth": bandwidth, "latency": latency})
                network_topology.append({"source": j, "destination": i, "bandwidth": bandwidth, "latency": latency})

    # Edge to Cloud Links
    for i in range(30, 45):
        for j in range(45, 50):
            if random.random() < 0.8:  # 80% chance to create a link
                bandwidth = random.randint(150, 200)
                latency = random.randint(30, 60)
                network_topology.append({"source": i, "destination": j, "bandwidth": bandwidth, "latency": latency})
                network_topology.append({"source": j, "destination": i, "bandwidth": bandwidth, "latency": latency})

    # Cloud to Cloud Links
    for i in range(45, 50):
        for j in range(i + 1, 50):
            if random.random() < 0.7:  # 70% chance to create a link
                bandwidth = random.randint(200, 300)
                latency = random.randint(61, 90)
                network_topology.append({"source": i, "destination": j, "bandwidth": bandwidth, "latency": latency})
                network_topology.append({"source": j, "destination": i, "bandwidth": bandwidth, "latency": latency})

    return network_topology

def main():
    hosts_config = generate_hosts()
    network_topology = generate_links()

    config = {
        "hosts": {
            "nb": 50,
            "configuration": hosts_config
        },
        "network": {
            "topology": network_topology
        }
    }

    with open('scaled_infrastructure.json', 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    main()
