from vizdoom import *
# import to do training
from tpg.trainer import Trainer,loadTrainer 
# import to run an agent (always needed)
from tpg.agent import Agent
import numpy as np
from PIL import Image
import random
import multiprocessing as mp
import time


def preprocessImg(img, size):

    img = np.rollaxis(img, 0, 3)    # It becomes (640, 480, 3)
#     print(img.shape)
    ing = img[15:-5,20:-20,:]
    img = Image.fromarray(img)
    img = img.resize(size)
#     img = np.array(img)/255
#     img = img.convert('1')

    return np.array(img)


def screen2state(screen_buffer):
    x_t = preprocessImg(screen_buffer, size=(img_rows, img_cols))
    s_t = np.stack(([x_t]*4), axis=2) # It becomes 64x64x4
    s_t = np.expand_dims(s_t, axis=0) # 1x64x64x4
    return s_t



def shape_reward(r_t, misc,prev_misc):

    if (misc[0] < prev_misc[0]): # Loss HEALTH
        r_t -= 1
    
    if misc[1] == 1:# Get ARMOR
        r_t +=500
    if misc[2] > prev_misc[2]:# Kill 
        r_t +=100
    if misc[-1] > prev_misc[-1]:# Move forward
        r_t +=1

    return r_t


def stack_frames(stacked_frames,state,is_new_episode):
    frame = preprocessImg(state,(img_rows,img_cols))
    if is_new_episode:
        stacked_frames = deque([frame for i in range(stack_size)],maxlen=4)
    else:
        stacked_frames.append(frame)
    stacked_state = np.stack(stacked_frames,0)
    return stacked_state, stacked_frames

def getState(inState):
    # each row is all 1 color
    
    rgbRows = np.reshape(inState,(len(inState[0])*len(inState), 3)).T
    return np.add(np.left_shift(rgbRows[0], 16),
        np.add(np.left_shift(rgbRows[1], 8), rgbRows[2]))





def runAgent(args):
    agent = args[0]
    envName = args[1]
    scoreList = args[2]
    numEpisodes = args[3] # number of times to repeat game
    numFrames = args[4] 
    
    
    
    
    


    
    stacked_frames = deque([np.zeros((108,124),dtype=np.int) for i in range(stack_size)],maxlen=4)
    
    # skip if task already done by agent
    if agent.taskDone(envName):
        print('Agent #' + str(agent.agentNum) + ' can skip.')
        scoreList.append((agent.team.id, agent.team.outcomes))
        return
    
#     env = gym.make(envName)

    game = DoomGame()
    game.load_config(envName)
    game.set_sound_enabled(False)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(False)
    game.init()
    game.new_episode()
    game_state = game.get_state()
    
    valActs = 4
    possible_actions = np.identity(valActs,dtype=np.float).tolist()

    scoreTotal = 0 # score accumulates over all episodes
    for ep in range(numEpisodes): # episode loop
        state = preprocessImg(game_state.screen_buffer,(img_rows,img_cols))
        scoreEp = 0
        numRandFrames = 0
        if numEpisodes > 1:
            numRandFrames = random.randint(0,30)
        prevals = game_state.game_variables
        for i in range(numFrames): # frame loop
            if i < numRandFrames:
                game.make_action(possible_actions[np.random.randint(0,valActs)])
                continue
            action = agent.act(getState(state))
            r = game.make_action(possible_actions[action])
            reward = 0
    
            game_state = game.get_state()
            is_terminated = game.is_episode_finished()
            
            # feedback from env
            
            if is_terminated:
                if r < 0:
                    reward = -100
                game.new_episode()
                game_state = game.get_state()
                
            vals = game_state.game_variables
            
            reward = shape_reward(reward,vals,prevals)
            prevals = vals
            state = preprocessImg(game_state.screen_buffer,(img_rows,img_cols))
            scoreEp += reward # accumulate reward in score
                
        print('Agent #' + str(agent.agentNum) + 
              ' | Ep #' + str(ep) + ' | Score: ' + str(scoreEp))
        scoreTotal += scoreEp
       
    scoreTotal /= numEpisodes
    game.close()
    agent.reward(scoreTotal, envName)
    scoreList.append((agent.team.id, agent.team.outcomes))




def train():    
    
    tStart = time.time()
    # stack_size=4
    envName = 'deadly_corridor.cfg'
    game = DoomGame()
    game.load_config(envName)
    game.set_sound_enabled(False)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(False)
    game.init()
    # acts = game.get_available_buttons_size()
    del game

    trainer = Trainer(actions=range(4), teamPopSize=30, rTeamPopSize=30)
    # trainer = loadTrainer('trainer.tn')
    # trainer = loadTrainer('trainer.tn')
    processes = 7
    man = mp.Manager()
    pool = mp.Pool(processes=processes, maxtasksperchild=1)

    allScores = [] # track all scores each generation

    for gen in range(100): # do 100 generations of training
        scoreList = man.list()

        # get agents, noRef to not hold reference to trainer in each one
        # don't need reference to trainer in multiprocessing
        agents = trainer.getAgents()# swap out agents only at start of generation

        # run the agents
        pool.map(runAgent, 
            [(agent, envName, scoreList, 1, 2000)
            for agent in agents])

        # apply scores, must do this when multiprocessing
        # because agents can't refer to trainer
        teams = trainer.applyScores(scoreList)
        # important to remember to set tasks right, unless not using task names
        # task name set in runAgent()
        trainer.evolve(tasks=[envName]) # go into next gen

        # an easier way to track stats than the above example
        scoreStats = trainer.fitnessStats
        allScores.append((scoreStats['min'], scoreStats['max'], scoreStats['average']))
        print('Time Taken (Hours): ' + str((time.time() - tStart)/3600))
        print('Gen: ' + str(gen))
        print('Results so far: ' + str(allScores))

    # clear_output()
    print('Time Taken (Hours): ' + str((time.time() - tStart)/3600))
    print('Results:\nMin, Max, Avg')
    for score in allScores:
        print(score[0],score[1],score[2])

    trainer.saveToFile('trainer.tn')



'''
testing
'''
def test():

    game = DoomGame()
    game.load_config("deadly_corridor.cfg")
    game.set_sound_enabled(True)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    # game.set_console_enabled(True)
    game.init()
    game.new_episode()
    game_state = game.get_state()



    action_size = game.get_available_buttons_size()
    # trainer = Trainer(actions=range(action_size),teamPopSize=20, rTeamPopSize=20) 
    trainer = loadTrainer('trainer.tn')
    possible_actions = np.identity(action_size,dtype=np.float).tolist()


    for gen in range(5): # generation loop
        curScores = [] # new list per gen

        agent = trainer.getAgents()[28]
        s_t = preprocessImg(game_state.screen_buffer,size=(img_rows, img_cols))
        while True: # loop to go through agents
    #         teamNum = len(agents)
    #         agent = agents.pop()
    #         print(agent)
            if agent is None:
                print('No agent')
                break # no more agents, so proceed to next gen

            state = s_t  # get initial state and prep environment
            prev_misc = game_state.game_variables
            for i in range(100): # run episodes 
   	# get action from agent
                action = agent.act(getState(state))
                # feedback from env

                game_state = game.get_state()
                state = preprocessImg(game_state.screen_buffer,size=(img_rows, img_cols))
                game.make_action(possible_actions[action])
                is_terminated = game.is_episode_finished()


                if is_terminated:
                    print('over!!')
                    game.new_episode()
                    game_state = game.get_state()
        trainer.evolve()
        print(sum(curScores)/len(curScores))

img_rows , img_cols = 64, 48
test()