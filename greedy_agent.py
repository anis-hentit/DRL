import numpy as np
from greedy_environment import GreedyEnvironment

class BestFitAgent:
    def choose_action(self, state, app_index):
        app = state['applications'][app_index]
        best_host_id = None
        best_host_resources = (float('inf'), float('inf'))  # Tuple to store (CPU, RAM)

        for comp_id, comp in app['components'].items():
            if not comp['deployed']:
                for host_id, host in state['hosts'].items():
                    if comp['CPU'] <= host['CPU'] and comp['RAM'] <= host['RAM']:
                        # Check if this host has fewer available resources than the current best one
                        if (host['CPU'], host['RAM']) < best_host_resources:
                            best_host_id = host_id
                            best_host_resources = (host['CPU'], host['RAM'])

                if best_host_id is not None:
                    print(f"Deployed for App{app_index} the comp_id {comp_id} on host {best_host_id}")
                    return (comp_id, best_host_id)

        return None  # No valid action found
