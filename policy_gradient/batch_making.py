from frames_stacking import stack_frames
import tensorflow as tf
import numpy as np
from environment_creation import create_environment

game, possible_actions = create_environment()

def make_batch(batch_size, stacked_frames, PGNetwork, tf_sess, state_size, stack_size):
    # Initialize lists: states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards
    states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards = [], [], [], [], []
    
    # Reward of batch is also a trick to keep track of how many timestep we made.
    # We use to to verify at the end of each episode if > batch_size or not.
    
    # Keep track of how many episodes in our batch (useful when we'll need to calculate the average reward per episode)
    episode_num  = 1
    
    # Launch a new episode
    game.new_episode()
        
    # Get a new state
    state = game.get_state().screen_buffer
    state, stacked_frames = stack_frames(stacked_frames, state, True, state_size)

    while True:
        # Run State Through Policy & Calculate Action
        action_probability_distribution = tf_sess.run(PGNetwork.action_distribution, 
                                                   feed_dict={PGNetwork.inputs_: state.reshape(1, *state_size)})
        
        # REMEMBER THAT WE ARE IN A STOCHASTIC POLICY SO WE DON'T ALWAYS TAKE THE ACTION WITH THE HIGHEST PROBABILITY
        # (For instance if the action with the best probability for state S is a1 with 70% chances, there is
        # 30% chance that we take action a2)
        action = np.random.choice(range(action_probability_distribution.shape[1]), 
                                  p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
        action = possible_actions[action]

        # Perform action
        reward = game.make_action(action)
        done = game.is_episode_finished()

        # Store results
        states.append(state)
        actions.append(action)
        rewards_of_episode.append(reward)
        
        if done:
            # The episode ends so no next state
            next_state = np.zeros((84, 84), dtype=np.int)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, stack_size)
            
            # Append the rewards_of_batch to reward_of_episode
            rewards_of_batch.append(rewards_of_episode)
            
            # Calculate gamma Gt
            discounted_rewards.append(discount_and_normalize_rewards(rewards_of_episode))
           
            # If the number of rewards_of_batch > batch_size stop the minibatch creation
            # (Because we have sufficient number of episode mb)
            # Remember that we put this condition here, because we want entire episode (Monte Carlo)
            # so we can't check that condition for each step but only if an episode is finished
            if len(np.concatenate(rewards_of_batch)) > batch_size:
                break
                
            # Reset the transition stores
            rewards_of_episode = []
            
            # Add episode
            episode_num += 1
            
            # Start a new episode
            game.new_episode()

            # First we need a state
            state = game.get_state().screen_buffer

            # Stack the frames
            state, stacked_frames = stack_frames(stacked_frames, state, True, stack_size)
         
        else:
            # If not done, the next_state become the current state
            next_state = game.get_state().screen_buffer
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, stack_size)
            state = next_state
                         
    return np.stack(np.array(states)), np.stack(np.array(actions)), np.concatenate(rewards_of_batch), np.concatenate(discounted_rewards), episode_num


def discount_and_normalize_rewards(episode_rewards, gamma):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
    
    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

    return discounted_episode_rewards