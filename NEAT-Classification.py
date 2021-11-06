from __future__ import print_function
import os
import neat
import numpy as np
import visualize
import time
import gym

env = gym.make("CartPole-v1")
env.reset()

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        env.reset()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        observation = env.reset()
        genome.fitness = 0
        for _ in range(200):
            # env.render()
            action = np.clip(round(net.activate(observation)[0]), 0, 1)
            observation, reward, done, info = env.step(action)
            genome.fitness += 1
            if done:
                env.close()
                break

def run(config_file):
    startTraining = time.time()
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-175')
    
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))
    
    # Run for up to 1000 generations
    completed = False
    generations = 0
    while not completed:
        winner = p.run(eval_genomes, 1000)
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        completed = rungame(winner_net, 100)
        p.generation += 1
    endTraining = time.time()
    trainingTime = endTraining - startTraining

    startTesting = time.time()
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    rungame(winner_net, 100)
    endTesting = time.time()
    testingTime = endTesting - startTesting

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    print("Training took ", trainingTime)
    print("Running 100 runs took ", testingTime)
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    node_names = {-1:'Cart Position', -2: 'Cart Velocity', -3:'Pole Angle', -4:'Pole Velocity', 0:'Push Direction'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


def rungame(net, runs):
    sumRewards = []
    for _ in range(runs):
        finalReward = 0
        observation = env.reset()
        for _ in range(200):
            # env.render()
            action = np.clip(round(net.activate(observation)[0]), 0, 1)
            observation, reward, done, info = env.step(action)
            finalReward += reward
            if done:
                break
        # print(finalReward)
        sumRewards.append(finalReward)
        env.close()

    if np.mean(np.array(sumRewards).flatten()) > 195:
        print("\nAverage score for " + str(runs) + " runs is " + str(np.mean(np.array(sumRewards).flatten())) + "\n")
        return True
    else:
        return False

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)


    # testgame()