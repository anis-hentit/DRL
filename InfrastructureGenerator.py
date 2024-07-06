import json
import random
import os

def generate_hosts(num_hosts):
    hosts_config = []
    # IoT Devices (0-29)
    for i in range(min(30, num_hosts)):
        hosts_config.append({"CPU": random.randint(1, 5), "BW": random.randint(1, 3)})

    # Edge Servers (30-44)
    for i in range(min(15, max(0, num_hosts - 30))):
        hosts_config.append({"CPU": random.randint(40, 60), "BW": random.randint(20, 40)})

    # Cloud Servers (45-49)
    for i in range(min(5, max(0, num_hosts - 45))):
        hosts_config.append({"CPU": random.randint(200, 300), "BW": random.randint(80, 120)})

    return hosts_config

def generate_links(num_hosts, density_factor):
    network_topology = []

    def should_create_link(probability):
        return random.random() < probability

    # IoT to IoT Links
    for i in range(min(30, num_hosts)):
        for j in range(i + 1, min(30, num_hosts)):
            if should_create_link(0.2 * density_factor):  # Adjust probability by density factor
                bandwidth = random.randint(30, 50)
                latency = random.randint(1, 10)
                network_topology.append({"source": i, "destination": j, "bandwidth": bandwidth, "latency": latency})
                network_topology.append({"source": j, "destination": i, "bandwidth": bandwidth, "latency": latency})

    # IoT to Edge Links
    for i in range(min(30, num_hosts)):
        for j in range(30, min(45, num_hosts)):
            if should_create_link(0.95 * density_factor):
                bandwidth = random.randint(50, 70)
                latency = random.randint(5, 15)
                network_topology.append({"source": i, "destination": j, "bandwidth": bandwidth, "latency": latency})
                network_topology.append({"source": j, "destination": i, "bandwidth": bandwidth, "latency": latency})

    # Edge to Edge Links
    for i in range(30, min(45, num_hosts)):
        for j in range(i + 1, min(45, num_hosts)):
            if should_create_link(0.8 * density_factor):
                bandwidth = random.randint(80, 100)
                latency = random.randint(16, 25)
                network_topology.append({"source": i, "destination": j, "bandwidth": bandwidth, "latency": latency})
                network_topology.append({"source": j, "destination": i, "bandwidth": bandwidth, "latency": latency})

    # Edge to Cloud Links
    for i in range(30, min(45, num_hosts)):
        for j in range(45, min(50, num_hosts)):
            if should_create_link(0.8 * density_factor):
                bandwidth = random.randint(150, 200)
                latency = random.randint(30, 60)
                network_topology.append({"source": i, "destination": j, "bandwidth": bandwidth, "latency": latency})
                network_topology.append({"source": j, "destination": i, "bandwidth": bandwidth, "latency": latency})

    # Cloud to Cloud Links
    for i in range(45, min(50, num_hosts)):
        for j in range(i + 1, min(50, num_hosts)):
            if should_create_link(0.90 * density_factor):
                bandwidth = random.randint(200, 300)
                latency = random.randint(61, 90)
                network_topology.append({"source": i, "destination": j, "bandwidth": bandwidth, "latency": latency})
                network_topology.append({"source": j, "destination": i, "bandwidth": bandwidth, "latency": latency})

    return network_topology

def main(num_hosts, density_factor):
    hosts_config = generate_hosts(num_hosts)
    network_topology = generate_links(num_hosts, density_factor)

    config = {
        "hosts": {
            "nb": num_hosts,
            "configuration": hosts_config
        },
        "network": {
            "topology": network_topology
        }
    }

    # Define the path to save the file in the /data directory
    output_dir = 'DRL/data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'scaled_infrastructure.json')

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {output_path}")

if __name__ == "__main__":
    main(50, 1.5)  # Example usage: 50 nodes, normal density
