from environment import FogEnvironment

def test_initialization():
    env = FogEnvironment()
    state = env.reset()  # Reset environment to initial state
    env.render()  # Render the current state

    print("Initialized hosts:", env.state['hosts'])
    print("Initialized components:", env.state['components'])
    print("Fixed positions:", env.state['fixed_positions'])
    print("Paths mapping per link:", env.state['paths'])

if __name__ == "__main__":
    test_initialization()
