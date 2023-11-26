import pygame
import numpy as np
from RacingEnv import RacingEnv
from ddqn_keras import DDQNAgent
import random

TOTAL_GAMETIME = 1000  # Max game time for one episode
N_EPISODES = 10000
REPLACE_TARGET = 50

game = RacingEnv()
game.fps = 60

ddqn_agent1 = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=5, epsilon=1.00, epsilon_end=0.10, epsilon_dec=0.9995,
                        replace_target=REPLACE_TARGET, batch_size=512, input_dims=19)
ddqn_agent2 = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=5, epsilon=1.00, epsilon_end=0.10, epsilon_dec=0.9995,
                        replace_target=REPLACE_TARGET, batch_size=512, input_dims=19)
ddqn_scores1 = []
ddqn_scores2 = []
eps_history = []

def run():
    for e in range(N_EPISODES):
        game.reset()  # Reset environment

        done = False
        score1 = 0
        score2 = 0
        counter1 = 0
        counter2 = 0
        action1 = 0
        action2 = 0
        observation_1, observation_2, reward, done = game.step(action1, action2)
        observation1 = np.array(observation_1)
        observation2 = np.array(observation_2)
        gtime = 0  # Set game time back to 0
        renderFlag = False 
        if e % 10 == 0 and e > 0:  # Render every 10 episodes
            renderFlag = True
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            action1 = ddqn_agent1.choose_action(observation1)
            action2 = ddqn_agent2.choose_action(observation2)
            observation_1, observation_2, reward, done = game.step(action1, action2)
            observation1 = np.array(observation_1)
            observation2 = np.array(observation_2)
            if reward == 0:
                counter1 += 1
                counter2 += 1
                if counter1 > 10 or counter2 > 10:
                    done = True
            else:
                counter1 = 0
                counter2 = 0

            score1 += reward
            score2 += reward

            ddqn_agent1.remember(observation1, action1, reward, observation_1, int(done))
            ddqn_agent2.remember(observation2, action2, reward, observation_2, int(done))

            observation1 = observation_1
            observation2 = observation_2

            if gtime % 5 == 0:  # Adjust the frequency as needed
                ddqn_agent1.learn()
                ddqn_agent2.learn()

            gtime += 1
            if gtime >= TOTAL_GAMETIME:
                done = True

            if renderFlag and e % 10 == 0:  # Adjust the frequency as needed
                game.render(action1, action2)

        eps_history.append(ddqn_agent1.epsilon)
        ddqn_scores1.append(score1)
        ddqn_scores2.append(score2)

        avg_score1 = np.mean(ddqn_scores1[max(0, e - 100):(e + 1)])
        avg_score2 = np.mean(ddqn_scores2[max(0, e - 100):(e + 1)])

        if e % REPLACE_TARGET == 0 and e > REPLACE_TARGET:
            ddqn_agent1.update_network_parameters()
            ddqn_agent2.update_network_parameters()

        if e % 10 == 0 and e > 10:
            ddqn_agent1.save_model()
            ddqn_agent2.save_model()
            print("save models")

        print('episode: ', e, 'score1: %.2f' % score1, 'score2: %.2f' % score2,
              ' average score1 %.2f' % avg_score1, ' average score2 %.2f' % avg_score2,
              ' epsilon1: ', ddqn_agent1.epsilon, ' epsilon2: ', ddqn_agent2.epsilon,
              ' memory size1', ddqn_agent1.memory.mem_cntr % ddqn_agent1.memory.mem_size,
              ' memory size2', ddqn_agent2.memory.mem_cntr % ddqn_agent2.memory.mem_size)
run()
