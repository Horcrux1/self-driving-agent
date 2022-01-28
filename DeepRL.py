class DQNAgent(nn.Module):
    def __init__(self):
        super().__init__()

        self.main_model = self.create_model()
        self.target_model = self.create_model()

        # copying the weight of the main model to the target model 
        self.target_model.load_state_dict(self.main_model.state_dict())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        
        
    def create_model(self, number_of_actions=NUM_OF_ACTIONS):
        "Creating a referenced model according to which the target and main models of the RL agent are made."
        # FIXME: if by any reason the following sequential layer is changed, the used dimensions are no longer valid
        model = nn.Sequential(
                    nn.Conv2d(3, 32, (8, 8), stride=4),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, (4, 4), stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, (3, 3), stride=1),
                    nn.ReLU(),
                    # in Pytorch for using linear layers after convs, one should manually crunch all the previous dimensions
                    nn.Flatten(),
                    # based on the following formula, dimensions of the convolved image is computed: (n+2p-f+1)//stride_size 
                    nn.Linear(64*7*7, 512),
                    nn.ReLU(),
                    # network's outputs are in accordance to the number of available actions taken by the agent 
                    nn.Linear(512, number_of_actions)
                )

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

        return model
    
    def updateReplayMemory(self, transition):
        "adds experience to the replay memory every step"
        # (current state, action, reward, new state, done)
        return self.replay_memory.append(transition)

    def getQs(self, state):
        " Queries main network for Q values given current observation space (environment state)"
        q_tensor = self.main_model(torch.tensor(state).view(-1, 3, 84, 84)/255)
        q_array = q_tensor.detach().numpy()
        return q_array


    def train(self, terminal_state, step, min_replay_memory_size=MIN_REPLAY_MEMORY_SIZE):
        "Trains main network every step during episode"
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < min_replay_memory_size:
            return 
        # Get a minibatch of random samples from memory replay buffer
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        
        # Get current states from minibatch, then query NN model for Q values
        current_states = torch.tensor([transition[0] for transition in minibatch]).view(-1, 3, 84, 84)/255
        current_qs_list = self.main_model(current_states).detach().numpy()
        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = torch.tensor([transition[3] for transition in minibatch]).view(-1, 3, 84, 84)/255
        future_qs_list = self.target_model(new_current_states).detach().numpy()

        X = []
        y = [] # target/desired output 

        # enumerate the batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q

            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch
        self.main_model.zero_grad() # zero the gradient buffers
        net_output = self.main_model(torch.tensor(X).view(-1, 3, 84, 84)/255)
        loss = self.loss_function(net_output, torch.tensor(y))
        loss.backward()
        self.optimizer.step() # does the parameter update 

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.main_model.state_dict())
            self.target_update_counter = 0

    def run(self, gameEnv, episodes=EPISODES, epsilon=epsilon, min_epsilon=MIN_EPSILON, epsilon_decay=EPSILON_DECAY, num_of_actions=NUM_OF_ACTIONS):
        "Running the agent by collecting experience step-wise and training the agent throughout the episodes"
        # Iterate over episodes
        for episode in tqdm(range(1, episodes + 1), ascii=True, unit='episodes', desc="Training the agent"):
            # Restarting episode - reset episode reward and step number
            step = 1
            episode_reward = 0
            # Reset environment and get initial observation
            current_state = gameEnv.reset()

            # Reset flag and start iterating until episode ends
            done = False
            while not done:
                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Get action from Q table
                    action = np.argmax(self.getQs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, num_of_actions)

                new_state, reward, done = gameEnv.step(action) 

                # TODO: log this variable through episodes 
                episode_reward += reward

                # Decay epsilon
                if epsilon > min_epsilon:  
                    epsilon *= epsilon_decay
                    epsilon = max(min_epsilon, epsilon)

                # Every step we update replay memory and train main network
                self.updateReplayMemory((current_state, action, reward, new_state, done))
                self.train(done, step)
                current_state = new_state
                step += 1