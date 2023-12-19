import os
from datetime import datetime, timedelta

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain.chains import create_extraction_chain
from langchain.prompts import (
    PromptTemplate,
)

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

nlu_prompt_text = """당신은 일정관리 시스템 입니다. 일정관리를 위해 필요한 slot, value를 추출합니다.

event_name이 확실하지 않을 땐 추출하지 않습니다.
time은 HH:MM 형식으로 출력합니다.
date는 YYYY-MM-DD 형식으로 출력합니다.

현재날짜: {today}
"""

dst_prompt_text = """당신은 일정관리 시스템 입니다. 일정관리를 위해 nlu_result를 분석하여 dialog_state를 업데이트 하세요.
업데이트 된 dialog_state에는 현재 dialog_sate db에 있는 값 중 slot, value 조건에 맞는 값만 남겨놓습니다.

# data
nlu_result: {nlu_result}
dialog_state: {dialog_state}
현재날짜: {today}

# 응답
- 업데이트된 dialog_state 를 dict 형태로 출력
"""

nlg_prompt_text = """당신은 일정관리 시스템의 Natural Language Generator 입니다. dialog_state를 분석하여 user에게 응답하세요.

# data
dialog_state: {dialog_state}
현재날짜: {today}

# 응답
- 자연어 형태로 출력
"""


class PromptAgent:
    def __init__(self, llm, verbose=False):
        self.nlu_chain = None
        self.dst_chain = None
        self.nlg_chain = None
        self.llm = llm
        self.verbose = verbose

        self.init_nlu_chain()
        self.init_dst_chain()
        self.init_nlg_chain()

        self.today = datetime.today().strftime("%Y-%m-%d %H:%M")

    def init_nlu_chain(self):
        schema = {
            "properties": {
                "event_name": {"type": "string"},
                "action": {"type": "string", "enum": ["create", "read", "update", "delete"]},
                "date": {"type": "string", "description": "날짜"},
                "time": {"type": "string", "description": "시간"},
            },
            "required": ["action"],
        }

        nlu_prompt = ChatPromptTemplate(
            messages=[
                AIMessagePromptTemplate.from_template(nlu_prompt_text),
                HumanMessagePromptTemplate.from_template("{user_input}"),
            ]
        )

        self.nlu_chain = create_extraction_chain(schema, self.llm, prompt=nlu_prompt, verbose=True)

    def run_nlu_chain(self, inp):
        response = self.nlu_chain.run({'user_input': inp, 'today': self.today})
        return response[0]

    def init_dst_chain(self):
        dst_prompt = ChatPromptTemplate(
            messages=[
                AIMessagePromptTemplate.from_template(dst_prompt_text),
                # HumanMessagePromptTemplate.from_template("{user_input}"),
            ],
            input_variables=["nlu_result", "dialog_state", "today"],
        )
        self.dst_chain = LLMChain(llm=self.llm, prompt=dst_prompt, verbose=True)

    def run_dst_chain(self, dialog_state, nlu_result):
        response = self.dst_chain.run({'dialog_state': dialog_state, 'nlu_result': nlu_result, 'today': self.today})
        return eval(response)

    def init_nlg_chain(self):
        nlg_prompt = ChatPromptTemplate(
            messages=[
                AIMessagePromptTemplate.from_template(nlg_prompt_text),
                # HumanMessagePromptTemplate.from_template("{user_input}"),
            ]
        )
        self.nlg_chain = LLMChain(llm=self.llm, prompt=nlg_prompt, verbose=True)

    def run_nlg_chain(self, dialog_state):
        response = self.nlg_chain.run({'dialog_state': dialog_state, 'today': self.today})
        return response


def main():
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a nice chatbot having a conversation with a human."
            ),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )

    # Notice that we `return_messages=True` to fit into the MessagesPlaceholder
    # Notice that `"chat_history"` aligns with the MessagesPlaceholder name
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Notice that we just pass in the `question` variables - `chat_history` gets populated by memory
    # response = conversation({"question": "hi"})

    # print(response)
    # conversation({"question": "안녕하세요."})
    # print(memory)

    schedule_prompt = PromptTemplate(
        template="""당신은 일정관리 시스템 입니다. 일정관리를 위해 필요한 slot, value를 추출합니다.
현재날짜: {today}

User Input: {user_input} 
        """,
        input_variables=["user_input", "today"],

    )

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    prompt_agent = PromptAgent(llm=llm, verbose=True)

    # chain.memory = memory

    dialog_state = {'event_name': '', 'action': '', 'date': '', 'time': '', 'db': []}

    # Input
    # inp = """내일 오후 10시에 산책가기 일정을 등록해줘"""
    inp = """내일 일정을 조회해줘"""

    nlu_result = prompt_agent.run_nlu_chain(inp=inp)
    system_action = nlu_result['action']

    date_today = datetime.today().strftime("%Y-%m-%d")
    date_tomorrow = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    schedule_list = [
        {
            'event_name': '산책가기',
            'date': date_today,
            'time': '10:00'
        },
        {
            'event_name': '데이트',
            'date': date_tomorrow,
            'time': '12:00'
        }
    ]
    if system_action == 'read':
        dialog_state.update({'db': schedule_list})

    print(f'dialog_state: {dialog_state}')

    dst_result = prompt_agent.run_dst_chain(dialog_state=dialog_state, nlu_result=nlu_result)
    dialog_state = dst_result

    nlg_result = prompt_agent.run_nlg_chain(dialog_state=dialog_state)
    print(nlg_result)

    exit()

    inp = """내일 오후에 친구와 약속이 있어"""
    response = nlu_chain.run({'user_input': inp, 'today': today})
    print(response)

    nlu_result = response[0]

    response = dst_chain.run({'dialog_state': dialog_state, 'nlu_result': nlu_result, 'today': today})

    print(response)

    # exit()

    # inp = """다음 주 수요일 10시 일정을 알려줘"""
    # response = chain.run({'user_input': inp, 'today': today})
    # print(response)
    #
    # inp = """나 내일 뭐해?"""
    # response = chain.run({'user_input': inp, 'today': today})
    # print(response)


if __name__ == '__main__':
    main()
