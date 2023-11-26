import pygame
import math
from Car import Car
from GameEnv import GOALREWARD, LIFE_REWARD, PENALTY, Ray, distance, myLine, myPoint, rotate, rotateRect
from Walls import Wall
from Walls import getWalls
from Goals import Goal
from Goals import getGoals
class RacingEnv:
    def __init__(self):
        pygame.init()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 20)
        self.fps = 120
        self.width = 1000
        self.height = 600
        self.history = []
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RACING DQN")
        self.screen.fill((0, 0, 0))
        self.back_image = pygame.image.load("track.png").convert()
        self.back_rect = self.back_image.get_rect().move(0, 0)
        self.action_space = None
        self.observation_space = None
        self.game_reward = 0
        self.score = 0
        self.reset()

    def reset(self):
        self.screen.fill((0, 0, 0))

        self.car1 = Car(35, 300, "car11.png")
        self.car2 = Car(75, 300, "car22.png")  # Adjust starting position for the second car
        self.walls = getWalls()
        self.goals = getGoals()
        self.game_reward = 0


    def step(self, action1, action2):
        done = False

        self.car1.action(action1)
        self.car2.action(action2)

        self.car1.update()
        self.car2.update()

        reward = LIFE_REWARD

        # Check if cars pass Goal and score
        for goal in self.goals:
            if self.car1.score(goal):
                goal.isactiv = False
                reward += GOALREWARD

            if self.car2.score(goal):
                goal.isactiv = False
                reward += GOALREWARD

        # Check if cars collided with walls
        for wall in self.walls:
            if self.car1.collision(wall) or self.car2.collision(wall):
                reward += PENALTY
                done = True

        # You might want to modify the state representation to include both cars' observations
        new_state_1 = self.car1.cast(self.walls)
        new_state_2 = self.car2.cast(self.walls)

        # Normalize states if needed
        if done:
            new_state_1 = None
            new_state_2 = None

        return new_state_1, new_state_2, reward, done

    def render(self, action1, action2):

        DRAW_WALLS = False
        DRAW_GOALS = False
        DRAW_RAYS = False

        pygame.time.delay(10)

        self.clock = pygame.time.Clock()
        self.screen.fill((0, 0, 0))

        self.screen.blit(self.back_image, self.back_rect)

        if DRAW_WALLS:
            for wall in self.walls:
                wall.draw(self.screen)
        
        if DRAW_GOALS:
            for goal in self.goals:
                goal.draw(self.screen)
                if goal.isactiv:
                    goal.draw(self.screen)

        self.car1.draw(self.screen)
        self.car2.draw(self.screen)

        if DRAW_RAYS:
            i = 0
            for pt in self.car.closestRays:
                pygame.draw.circle(self.screen, (0,0,255), (pt.x, pt.y), 5)
                i += 1
                if i < 15:
                    pygame.draw.line(self.screen, (255,255,255), (self.car.x, self.car.y), (pt.x, pt.y), 1)
                elif i >=15 and i < 17:
                    pygame.draw.line(self.screen, (255,255,255), ((self.car.p1.x + self.car.p2.x)/2, (self.car.p1.y + self.car.p2.y)/2), (pt.x, pt.y), 1)
                elif i == 17:
                    pygame.draw.line(self.screen, (255,255,255), (self.car.p1.x , self.car.p1.y ), (pt.x, pt.y), 1)
                else:
                    pygame.draw.line(self.screen, (255,255,255), (self.car.p2.x, self.car.p2.y), (pt.x, pt.y), 1)
        # car 1
        # score
        text_surface1 = self.font.render(f'Points {self.car1.points}', True, pygame.Color('red'))
        self.screen.blit(text_surface1, dest=(0, 0))
        # speed
        text_surface1 = self.font.render(f'Speed {self.car1.vel*-1}', True, pygame.Color('red'))
        self.screen.blit(text_surface1, dest=(800, 0))
        # car2 
        # score
        text_surface2 = self.font.render(f'Points {self.car2.points}', True, pygame.Color('yellow'))
        self.screen.blit(text_surface2, dest=(0, 40))
        # speed
        text_surface2 = self.font.render(f'Speed {self.car2.vel*-1}', True, pygame.Color('yellow'))
        self.screen.blit(text_surface2, dest=(800, 40))

        self.clock.tick(self.fps)
        pygame.display.update()

    def close(self):
        pygame.quit()

# def main():
#     env = RacingEnv()

#     # Run the main loop
#     running = True
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False

#         # Example: You can control the first car with arrow keys
#         keys = pygame.key.get_pressed()
#         action1 = 0
#         if keys[pygame.K_UP]:
#             action1 = 1
#         elif keys[pygame.K_DOWN]:
#             action1 = 4
#         elif keys[pygame.K_LEFT]:
#             action1 = 2
#         elif keys[pygame.K_RIGHT]:
#             action1 = 3

#         # Example: You can control the second car with W, A, S, D keys
#         action2 = 0
#         if keys[pygame.K_w]:
#             action2 = 1
#         elif keys[pygame.K_s]:
#             action2 = 4
#         elif keys[pygame.K_a]:
#             action2 = 2
#         elif keys[pygame.K_d]:
#             action2 = 3

#         # Perform a step in the environment
#         state, reward, done = env.step(action1, action2)

#         # Render the current state of the environment
#         env.render(action1, action2)

#         # Check if the user closes the window
#         if done:
#             running = False

#     # Close the environment when the loop ends
#     env.close()

# if __name__ == "__main__":
#     main()

