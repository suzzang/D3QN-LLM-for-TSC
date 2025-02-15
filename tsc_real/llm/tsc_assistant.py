import numpy as np
from typing import List
from loguru import logger

from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationTokenBufferMemory
from  langchain.chains.conversation.base import ConversationChain
from langchain.prompts import ChatPromptTemplate   #导入聊天提示模板
from langchain.chains.llm import LLMChain   #导入LLM链。

from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

from llm.tsc_agent_prompt import SYSTEM_MESSAGE_SUFFIX
from llm.tsc_agent_prompt import (
    SYSTEM_MESSAGE_SUFFIX,
    SYSTEM_MESSAGE_PREFIX,
    HUMAN_MESSAGE,
    FORMAT_INSTRUCTIONS,
    TRAFFIC_RULES,
    DECISION_CAUTIONS,
    HANDLE_PARSING_ERROR
)

class TSCAgent:
    def __init__(self, 
                 llm:ChatOpenAI, 
                 verbose:bool=True,state:float=[] ) -> None:
        self.tls_id='J1'
        self.llm = llm # ChatGPT Model
        self.tools = [] 
        self.state= state
        self.memory = ConversationTokenBufferMemory(llm=self.llm, max_token_limit=2048)
        memory = ConversationBufferMemory()
        self.assessment = ConversationChain( llm=llm, memory = memory, verbose=True )
        self.phase2movements={
                        "Action 0": ["E0_s","-E1_s"],
                        "Action 1": ["E0_l","-E1_l"],
                        "Action 2": ["-E3_s","-E2_s"],
                        "Action 3": ["-E3_l","-E2_l"],
                    } 
        self.movement_ids=["E0_s","-E1_s","-E1_l","E0_l","-E3_s","-E2_s","-E3_l","-E2_l"]     

    def get_phase(self):
        phases_4=[[1, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 1]]
        
        return np.array(phases_4)
   
   
    def agent_run(self, sim_step:float, weight:int=0, obs:float=[], infos: list={}):
       
        logger.info(f"SIM: Decision at step {sim_step} is running:")
        current_weight = weight

        Occupancy=infos['occupancy']
        queue_length_per_movement=infos['jam_length_meters']
        waiting_time_per_movement=infos['waiting_time']

        review_template = """
        decision: As the reward weight decider, let me know whether I should set the weight 'alpha' higher or lower. Note that values below 0.5 are considered lower, and values above 0.5 are considered higher.
        explanations: Provide detailed analysis on how each observation (e.g., occupancy, queue length on each movement, waiting time on each movement, travel time) affects the reward function. Adjust the weight 'alpha' to find the optimal balance between minimizing the standard deviation of waiting time on each movement and queue length on each movement. Your goal is to maximize the reward function, which is designed to optimize traffic flow.
        final_weight: Propose the new value for the weight 'alpha' (between 0 and 1) in the reward function, focusing on minimizing the standard deviation of both waiting time and queue length on each movement.

        The current reward function is:
        R = -[alpha * std(waiting_time_per_movement) + (1-alpha) * std(queue_length_per_movement)]
        The value of 'alpha' should be between 0 and 1. Your goal is to maximize this reward function while minimizing the variability (standard deviation) in waiting time and queue length on each movement to ensure smoother traffic flow.

        Format the output as JSON with the following keys:
        decision
        explanations
        final_weight

        observation: {observation}
        {format_instructions}
        """

        prompt = ChatPromptTemplate.from_template(template=review_template)

        decision = ResponseSchema(name="decision",
                                description="As the reward weight decider, let me know whether I should set the weight 'alpha' higher or lower.")
        explanations = ResponseSchema(name="explanations",
                                    description="Provide detailed analysis on how each observation (e.g., occupancy, queue length on each movement, waiting time on each movement, travel time) affects the reward function. Adjust the weight 'alpha' to find the optimal balance between minimizing the standard deviation of waiting time on each movement and queue length on each movement. Your goal is to maximize the reward function, which is designed to optimize traffic flow.")
        final_weight = ResponseSchema(name="final_weight",
                                    description="Propose the new value for the weight 'alpha' (between 0 and 1) in the reward function, focusing on minimizing the standard deviation of both waiting time and queue length on each movement.")

        response_schemas = [decision, 
                    explanations,
                    final_weight]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        
        
        observation = (f"""
        You are tasked with determining the weight 'alpha' in the reinforcement learning reward function. 
        The current situation involves controlling the traffic signal at intersection with ID`{self.tls_id}`. 
        Your goal is to adjust the weight 'alpha' to optimize the traffic flow based on the current state of the intersection.

        The current Weight is: `{current_weight}`.
        The vehicles' occupancy of each movement is: `{Occupancy}`.
        The number of cars waiting in each movement (queue length) is: `{queue_length_per_movement}`.
        The waiting time on each movement is as follows: `{waiting_time_per_movement}`.
        Phase to Movement mapping: '{self.phase2movements}'.
        
        The current reward function is as follows:
        R = -[alpha * std(waiting_time_per_movement) + (1-alpha) * std(queue_length_per_movement)]
        The value of 'alpha' should be between 0 and 1. Your goal is to maximize this reward function by minimizing the variability (standard deviation) in both waiting time and queue length on each movement.
        
        """)


        messages = prompt.format_messages(observation=observation, format_instructions=format_instructions)
        print(messages[0].content)
        logger.info('RL:'+messages[0].content) #加入RL解析 输入到日志文件
        r = self.llm(messages)
        output_dict = output_parser.parse(r.content)
        print(r.content)
        logger.info('RL_content:'+r.content)
        final_weight = output_dict.get('final_weight')
        logger.info('RL_final_weight:'+final_weight)
        print('-'*10)
        final_weight=float(final_weight)
        return final_weight
    

    def ask_adjust_weight(self, episode:int,weight:float, epsilon:float, rewards:list={}):
        current_episode = episode
        current_weight = weight
        episode_rewards = rewards

        ask_template = """
        Your role is to determine whether the weight 'alpha' in the reinforcement learning reward function should be adjusted to improve traffic flow.
        Please note that the rewards are intentionally set to be negative, and a less negative reward (i.e., closer to zero) indicates better performance. 
        Therefore, the goal is to make the rewards less negative over time, indicating improved traffic flow.
        
        The epsilon (ε) value represents the balance between exploration (trying new actions) and exploitation (using learned knowledge). 
        The initial epsilon value was 1.0, and it will decay down to a minimum value of 0.3 over time. The current epsilon value is the average across the episode, reflecting the overall balance of exploration and exploitation during this episode.


        decision:Decide whether the weight 'alpha' should be adjusted in this episode.Return 1 if the decision is 'Yes', and return 0 if the decision is 'No'.
        explanations: Provide the reasoning for your decision based on the provided data.
        
        The current reward function is:
        R = -[alpha * std(waiting_time_per_movement) + (1-alpha) * std(queue_length_per_movement)]
       

        Format the output as JSON with the following keys:
        decision
        explanations

        informations: {informations}
        {format_instructions}
        """

        prompt = ChatPromptTemplate.from_template(template=ask_template)

        decision = ResponseSchema(name="decision",
                                description="Decide whether the weight 'alpha' should be adjusted in this episode.Return 1 if the decision is 'Yes', and return 0 if the decision is 'No'.")
        explanations = ResponseSchema(name="explanations",
                                    description="Provide the reasoning for your decision based on the provided data.")

        response_schemas = [decision, 
                    explanations]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()


        informations = (f"""
        You are responsible for deciding whether the weight 'alpha' in the reinforcement learning reward function should be adjusted. 
        The current task involves controlling the traffic signal at intersection ID `{self.tls_id}` using the D3QN (Dueling Double Deep Q-Network) algorithm.

        Your goal is to determine, based on the information provided below, whether the weight 'alpha' should be adjusted in this episode to improve traffic flow control.

        Here is the relevant information:
        - The Total Episode number for training is 301
        - Current Episode number: `{current_episode}`
        - Current Weight 'alpha': `{current_weight}`
        - Average epsilon value (episodes 0 to {current_episode - 1}): `{epsilon}`
        - Episode rewards (0 to {current_episode - 1}): `{episode_rewards}`
        - The current reward function is as follows:
        R = -[alpha * std(waiting_time_per_movement) + (1-alpha) * std(queue_length_per_movement)]

        Based on this information, decide if the weight 'alpha' should be adjusted. 
        Provide 'Yes' if an adjustment is needed, or 'No' if the current weight should be maintained.
        """)

        
        messages = prompt.format_messages(informations=informations,format_instructions=format_instructions)
        print(messages[0].content)
        logger.info('RL:'+messages[0].content)
        r = self.llm(messages)
        output_dict = output_parser.parse(r.content)
        print(r.content)
        logger.info('RL_content:'+r.content)
        decision = output_dict.get('decision')
        logger.info('RL_decision:'+str(decision))
        print('-'*10)
        decision=int(decision)
        return decision
    