import torch
from torch import nn
from torch.distributions.categorical import Categorical
import tqdm

learning_rate = 1e-3
num_runs = 10

#Takes previous input and descision from agent and makes one run
def simple_run(previous_state = None, model_output = None, batch_size = 8, num_runs = 3):
    init_dice = torch.randint(low = 1, high = 7, size = (batch_size,5))
    # print(init_dice)
    # print(model_output)
    # print(previous_state)

    if previous_state is not None and model_output is not None:
        input = previous_state[:, :-1]
        mask = model_output.ge(1) #Greater or equal 1
        input[mask] = init_dice[mask] #WILL BE AFFECTED BY ADDDING BATCH-DIM
        return torch.cat((input, previous_state[:, -1:]-1), dim = 1)  #WILL BE AFFECTED BY ADDDING BATCH-DIM
    else: 
        input = init_dice
        return torch.cat((input, torch.tensor([[num_runs]]).repeat((batch_size,1))), dim = 1)

#Testing the function: IT seems like its working
# print(simple_run())
# print("-------")
# print(simple_run(previous_state=torch.tensor([[2, 2, 2, 2, 2, 1]]).repeat((8,1)), model_output=torch.tensor([[0, 0, 0, 0, 0]]).repeat((8,1))))
# print("-------")

#Starting of with a simple MLP with a RELU 
class simple_net(nn.Module):

    def __init__(self, hidden_dim, num_embeddings = 6, num_dice = 5):
        super().__init__()
        self.num_dice = num_dice
        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=num_dice)
        self.input_layer = nn.Linear(num_dice*self.embed.embedding_dim+1, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 2*num_dice)
        self.ReLU = nn.ReLU()
    def forward(self, x): # Consists of 5 ints (1-6) representing dice rolls and 1 final int (>0) representing the number of turns left (1, 6)
        embeddings = self.embed(x[:, :-1]-1)
        embeddings_flattened = embeddings.view(-1, self.embed.embedding_dim*self.num_dice)
        # print(embeddings_flattened.shape)
        # print(x[:, -1:].shape)
        x = torch.cat((embeddings_flattened, x[:, -1:]), dim = 1) #Stack emeddings and append number of rolls left.  -1: means makes sure to keep first dimension
        # print(x.shape)
        x = self.input_layer(x)
        x = self.ReLU(x)
        x = self.output_layer(x)
        x = x.view(-1, self.num_dice, 2)
        return x


class yazi_chance_agent():

    def __init__(self, policy_network):
        self.policy_network = policy_network
        self.softmax = torch.nn.Softmax(dim=2)

    def __call__(self, state, strategy = 'sample'):
        logits = self.policy_network(state)
        logits = self.softmax(logits)

        if strategy == "greedy":
            return logits.argmax(dim = 2)
        elif strategy == 'sample':
            m = Categorical(logits)
            output = m.sample()
            return output, m.log_prob(output)
        else:
            print("No such strategy...")


def calc_scores(states):
    return states[:, :-1].sum(dim=1)-15

def run_round(agent, batch_size = 8):
    all_states = []
    all_actions = []
    all_log_probs = []
    
    states = simple_run(batch_size=8, num_runs=num_runs)
    all_states.append(states)
    
    for run in range(num_runs):
        actions, log_probs = agent(states)
        states = simple_run(previous_state=states, model_output=actions)

        all_actions.append(actions)
        all_log_probs.append(log_probs)
        all_states.append(states)
    
    return all_actions, all_log_probs, all_states, calc_scores(states)


def training_loop():
    net = simple_net(hidden_dim = 20, num_embeddings=6, num_dice=5)
    agent = yazi_chance_agent(policy_network=net)



    optimizer = torch.optim.AdamW(agent.policy_network.parameters(), lr=learning_rate)
    losses = []

    for i in tqdm.tqdm(range(100000)):
        all_actions, all_log_probs, all_states, rewards = run_round(agent)

        loss = 0
        for tensor in all_log_probs:
            loss += (-tensor.T*rewards).T        
        loss = loss.mean()
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return agent, losses

agent, losses = training_loop()



###Plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

# Plot the losses
ax.plot(losses, label='Loss')

# Add a title and labels
ax.set_title('Loss Over Time')
ax.set_xlabel('Iterations')
ax.set_ylabel('Loss')

# Add a legend
ax.legend()

# Show the plot
plt.show()
