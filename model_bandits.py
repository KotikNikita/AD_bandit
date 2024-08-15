import torch
import torch.nn as nn
import numpy as np
# from scipy import stats as st
import matplotlib.pyplot as plt
from bandit import BernoulliMultiArmedBandit_TaskDistribution
from utils import averaging_reward
# import wandb
# from torch.nn import functional as F
# from torch.utils.data import DataLoader, IterableDataset
from typing import Optional
from torch.nn import functional as F
from tqdm import tqdm
# from tqdm.auto import trange


class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]

        norm_x = self.norm1(x)
        attention_out = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 10,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        max_action: float = 1.0,
        possible_actions: int = 10,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.action_head = nn.Linear(embedding_dim, possible_actions)
        # nn.Sequential(nn.Linear(embedding_dim, action_dim), nn.Tanh())
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action
        self.possible_actions = possible_actions

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
        returns_to_go: torch.Tensor,  # [batch_size, seq_len]
        time_steps: torch.Tensor,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states) + time_emb
        act_emb = self.action_emb(actions) + time_emb
        returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb
        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        sequence = (
            torch.stack([returns_emb, state_emb, act_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )
        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )
        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        # [batch_size, seq_len, action_dim]
        # predict actions only from state embeddings
        out = self.action_head(out[:, 1::3]) * self.max_action
        return out


def train(model,
          dataset,
          subsequence_length: int = 148,
          num_epochs: int = 1500,
          batch_size: int = 32,
          clip_grad: float = 0.25,
          env_distr: BernoulliMultiArmedBandit_TaskDistribution = BernoulliMultiArmedBandit_TaskDistribution(6),
          PATH: str = './model_name.pt',
          best_loss: int = 1000,
          ):
    losses = []
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=3.1e-4, betas=(0.68, 0.999), weight_decay=4.4e-03)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda steps: min((steps + 1) / 10, 1))
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    for epoch in tqdm(range(num_epochs)):
        indexes = list(np.random.choice(np.arange(len(dataset)), size=batch_size))
        chosen_histories = [dataset[i] for i in indexes]
        j = np.random.randint(low=0, high=len(chosen_histories[0]["observations"]) - subsequence_length, size=batch_size, dtype=int)
        states = np.stack([history["observations"][j[i]:j[i]+subsequence_length] for (i, history) in enumerate(chosen_histories)])
        actions = np.stack([history["actions"][j[i]:j[i]+subsequence_length] for (i, history) in enumerate(chosen_histories)])
        rewards = np.stack([history["rewards"][j[i]:j[i]+subsequence_length] for (i, history) in enumerate(chosen_histories)])
        if len(states.shape) < 3:
            states = torch.Tensor(states).unsqueeze(-1)
        if len(actions.shape) < 3:
            actions = torch.Tensor(actions).unsqueeze(-1)
        rewards = torch.Tensor(rewards)
        time_steps = torch.zeros((batch_size, subsequence_length), dtype=torch.long)
        mask = np.ones((states.shape[0], states.shape[1]))
        mask = torch.Tensor(mask)
        padding_mask = ~mask.to(torch.bool)
        output = model(states, actions, rewards, time_steps, padding_mask)
        loss = F.cross_entropy(
                        input=output.flatten(0, 1),
                        target=actions.flatten(0).to(torch.long),
                        )
        if loss <= best_loss:
            best_loss = loss.detach()
            torch.save(model, PATH)
        optim.zero_grad()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optim.step()
        scheduler.step()
        losses.append(loss.detach().numpy())

        if epoch % 100 == 0:
            model.eval()
            plt.plot(np.arange(len(losses)), losses)
            plt.title("Loss")
            plt.show()

            # num_runs = 5
            # # num_episodes = 300
            # rewards = [[] for i in range(num_runs)]
            # actionss = [[] for i in range(num_runs)]
            # pss = []
            # running_mean_rewards = []
            # for i in range(num_runs):
            #     env = env_distr.sample_task()
            #     # print(f'ps of env number {    i} is {env.ps}')
            #     pss.append(env.ps)
            #     states = torch.zeros(
            #         1, model.episode_len + 1, model.state_dim, dtype=torch.float)
            #     actions = torch.zeros(
            #         1, model.episode_len, model.action_dim, dtype=torch.float)
            #     returns = torch.zeros(1, model.episode_len + 1, dtype=torch.float)
            #     time_steps = torch.arange(model.episode_len, dtype=torch.long)
            #     states[:, 0] = torch.as_tensor(0)
            #     RTG = model.episode_len
            #     # returns[:, 0] = torch.as_tensor(RTG) #  0
            #     returns[:, 0] = torch.as_tensor(RTG)
            #     time_steps = time_steps.view(1, -1)
            #     for step in range(model.episode_len):
            #         predicted_actions = model(
            #             states[:, : step + 1][:, -model.seq_len :],
            #             actions[:, : step + 1][:, -model.seq_len :],
            #             returns[:, : step + 1][:, -model.seq_len :],
            #             time_steps[:, : step + 1][:, -model.seq_len :],
            #         )
            #         # print(f'shape is {predicted_actions.shape}')
            #         # predicted_action = predicted_actions[0, -1].argmax().detach().numpy()
            #         predicted_action = np.random.choice(np.arange(k),size=1,p=F.softmax(predicted_actions[0, -1]).detach().numpy())
            #         # print(f'predicted_actions.shape = {predicted_actions.shape}')
            #         next_state, reward = 0, env.run(predicted_action)
            #         rewards[i].append(reward) #  reward
            #         actionss[i].append(predicted_action)

            #         actions[:, step] = torch.as_tensor(predicted_action)
            #         states[:, step + 1] = torch.as_tensor(next_state)
            #         # RTG -= 1
            #         # RTG -= (1 - reward)
            #         # returns[:, step + 1] = torch.as_tensor(RTG) #  reward
            #         returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)

            # rewards = [np.array(rewardss) for rewardss in rewards]
            # for i in range(num_runs):
            #     running_mean_rewards.append(averaging_reward(rewards[i]))
            # for i in range(num_runs):
            #     plt.plot(np.arange(len(running_mean_rewards[i])), running_mean_rewards[i])
            #     plt.hlines(y=pss[i].max(), xmin=0, xmax=len(running_mean_rewards[i]), linewidth=2, color='r')
            #     # plt.ylim([3, 6])
            #     plt.xlabel('episodes')
            #     plt.ylabel('averaged reward')
            #     plt.grid(True)
            #     plt.show()

            model.train()
    return losses


def eval_model(model,
               num_runs: int = 200,
               env_distr: BernoulliMultiArmedBandit_TaskDistribution = BernoulliMultiArmedBandit_TaskDistribution(6),
               k: int = 6
               ):
    rewards = [[] for i in range(num_runs)]
    actionss = [[] for i in range(num_runs)]
    pss = []
    running_mean_rewards = []
    for i in tqdm(range(num_runs)):
        env = env_distr.sample_task()
        # print(f'ps of env number {i} is {env.ps}')
        pss.append(env.ps)
        states = torch.zeros(
            1, model.episode_len + 1, model.state_dim, dtype=torch.float)
        actions = torch.zeros(
            1, model.episode_len, model.action_dim, dtype=torch.float)
        returns = torch.zeros(1, model.episode_len + 1, dtype=torch.float)
        time_steps = torch.arange(model.episode_len, dtype=torch.long)
        states[:, 0] = torch.as_tensor(0)
        returns[:, 0] = torch.as_tensor(model.episode_len)
        time_steps = time_steps.view(1, -1)
        # print(f'rewards = {rewards}')
        for step in range(model.episode_len):
            predicted_actions = model(
                states[:, : step + 1][:, -model.seq_len:],
                actions[:, : step + 1][:, -model.seq_len:],
                returns[:, : step + 1][:, -model.seq_len:],
                time_steps[:, : step + 1][:, -model.seq_len:],
            )
            predicted_action = np.random.choice(np.arange(k), size=1, p=F.softmax(predicted_actions[0, -1]).detach().numpy())
            next_state, reward = 0, env.run(predicted_action)
            rewards[i].append(reward)
            actionss[i].append(predicted_action)

            actions[:, step] = torch.as_tensor(predicted_action)
            states[:, step + 1] = torch.as_tensor(next_state)
            returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)

    rewards = [np.array(rewardss) for rewardss in rewards]
    for i in range(num_runs):
        running_mean_rewards.append(averaging_reward(rewards[i]))
    last_returns = np.array([reward[-1] for reward in running_mean_rewards])
    median = np.median(last_returns)
    print(f'median = {median}')
    return running_mean_rewards
