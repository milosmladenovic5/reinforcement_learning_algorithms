import tensorflow as tf
import numpy as np

from environment_creation import create_environment
from frames_stacking import stack_frames

class GameBatch:
    def __init__(self, batch_size, stacked_frames, PGNetwork, tf_sess, state_size, stack_size):
        self.batch_size = batch_size
        self.stacked_frames = stacked_frames
        self.PGNetwork = PGNetwork
        self.tf_sess =  tf_sess
        self.state_size = state_size
        self.stack_size = stack_size
        
        game, possible_actions = create_environment()

        self.game = game
        self.possible_actions = possible_actions

    def make_batch(self, gamma):
        # Initialize lists: states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards
        states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards = [], [], [], [], []
        
        # Reward of batch is also a trick to keep track of how many timestep we made.
        # We use to to verify at the end of each episode if > batch_size or not.
        
        # Keep track of how many episodes in our batch (useful when we'll need to calculate the average reward per episode)
        episode_num  = 1
        
        # Launch a new episode
        self.game.new_episode()
            
        # Get a new state
        state = self.game.get_state().screen_buffer
        state, stacked_frames = stack_frames(self.stacked_frames, state, True, self.state_size)

        while True:
            # Run State Through Policy & Calculate Action
            action_probability_distribution = self.tf_sess.run(self.PGNetwork.action_distribution, 
                                                    feed_dict={self.PGNetwork.inputs_: state.reshape(1, *self.state_size)})
            
            # REMEMBER THAT WE ARE IN A STOCHASTIC POLICY SO WE DON'T ALWAYS TAKE THE ACTION WITH THE HIGHEST PROBABILITY
            # (For instance if the action with the best probability for state S is a1 with 70% chances, there is
            # 30% chance that we take action a2)
            action = np.random.choice(range(action_probability_distribution.shape[1]), 
                                    p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
            action = self.possible_actions[action]

            # Perform action
            reward = self.game.make_action(action)
            done = self.game.is_episode_finished()

            # Store results
            states.append(state)
            actions.append(action)
            rewards_of_episode.append(reward)
            
            if done:
                # The episode ends so no next state
                next_state = np.zeros((84, 84), dtype=np.int)
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, self.stack_size)
                
                # Append the rewards_of_batch to reward_of_episode
                rewards_of_batch.append(rewards_of_episode)
                
                # Calculate gamma Gt
                discounted_rewards.append(self.discount_and_normalize_rewards(rewards_of_episode, gamma))
            
                # If the number of rewards_of_batch > batch_size stop the minibatch creation
                # (Because we have sufficient number of episode mb)
                # Remember that we put this condition here, because we want entire episode (Monte Carlo)
                # so we can't check that condition for each step but only if an episode is finished
                if len(np.concatenate(rewards_of_batch)) > self.batch_size:
                    break
                    
                # Reset the transition stores
                rewards_of_episode = []
                
                # Add episode
                episode_num += 1
                
                # Start a new episode
                self.game.new_episode()

                # First we need a state
                state = self.game.get_state().screen_buffer

                # Stack the frames
                state, stacked_frames = stack_frames(stacked_frames, state, True, self.stack_size)
            
            else:
                # If not done, the next_state become the current state
                next_state = self.game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, self.stack_size)
                state = next_state
                            
        return np.stack(np.array(states)), np.stack(np.array(actions)), np.concatenate(rewards_of_batch), np.concatenate(discounted_rewards), episode_num


    def discount_and_normalize_rewards(self, episode_rewards, gamma):
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        cumulative = 0.0
        for i in reversed(range(len(episode_rewards))):
            cumulative = cumulative * gamma + episode_rewards[i]
            discounted_episode_rewards[i] = cumulative
        
        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards)
        discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

        return discounted_episode_rewards