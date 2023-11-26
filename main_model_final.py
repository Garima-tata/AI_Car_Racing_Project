import pygame
import numpy as np
from RacingEnv import RacingEnv
from ddqn_keras import DDQNAgent

TOTAL_GAMETIME = 10000
N_EPISODES = 10000
REPLACE_TARGET = 10

game = RacingEnv()
game.fps = 60

GameTime = 0 
GameHistory = []
renderFlag = True

ddqn_agent1 = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=5, epsilon=0.02, epsilon_end=0.01, epsilon_dec=0.999, replace_target= REPLACE_TARGET, batch_size=64, input_dims=19,fname='ddqn_model_main.h5')
ddqn_agent2 = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=5, epsilon=0.02, epsilon_end=0.01, epsilon_dec=0.999, replace_target= REPLACE_TARGET, batch_size=64, input_dims=19,fname='ddqn_model_main.h5')

ddqn_agent1.load_model()
ddqn_agent1.update_network_parameters()
ddqn_agent2.load_model()
ddqn_agent2.update_network_parameters()
ddqn_scores = []
eps_history = []
def run():
    #scores = deque(maxlen=100)
    for e in range(N_EPISODES):
        #reset env 
        game.reset()
        done = False
        done = False
        score1 = 0
        score2 = 0
        counter1 = 0
        counter2 = 0
        gtime = 0
        #first step
        observation_1, observation_2, reward, done = game.step(0, 0)
        observation1 = np.array(observation_1)
        observation2 = np.array(observation_2)
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    run = False
                    return
            action1 = ddqn_agent1.choose_action(observation1)
            action2 = ddqn_agent2.choose_action(observation2)
            observation_1,observation_2, reward, done = game.step(action1, action2)
            observation1 = np.array(observation_1)
            observation2 = np.array(observation_2)
            if reward == 0:
                counter1 += 1
                counter2 += 1
                if counter1 > 100 or counter2 > 100:
                    done = True
            else:
                counter1 = 0
                counter2 = 0
            score1 += reward
            score2 += reward
            observation1 = observation_1
            observation2 = observation_2
            if(score1 > 100):
                # render on the pygame window that car 1 wins
                print("car 1 wins")
                done = True
                return
            elif(score2 > 100):
                # render that car 2 wins
                print("car 2 wins")
                done = True
                return
            gtime += 1
            if gtime >= TOTAL_GAMETIME:
                done = True
            if renderFlag and e % 10 == 0:
                game.render(action1, action2)         

run()        