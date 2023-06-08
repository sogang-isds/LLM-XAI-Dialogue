import logging
import os

import gradio as gr
import numpy as np
import openai
import pandas as pd
from langchain.chat_models import ChatOpenAI
import hydra
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer, util

chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo")

openai.api_key = os.getenv("OPENAI_API_KEY")

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

messages = []


def request_chatgpt(prompt, chat_history):
    messages.append(HumanMessage(content=prompt))
    if prompt.startswith('debug'):
        response_message = f'echo - {prompt}'
    else:
        ai_message = chat_llm(messages)
        messages.append(ai_message)
        response_message = ai_message.content
    chat_history.append((prompt, response_message))

    return "", chat_history


def get_question(sentences):
    question = ""
    for sentence in sentences:
        print(sentence)
        if sentence.startswith("질문:"):
            question = sentence
            break
    return question


def click_button(msg):
    sentences = msg.split('\n')
    question = get_question(sentences)

    print('question: ', question)


class XAILaw:
    def __init__(self, knowlegde_file):
        self.data = pd.read_csv(knowlegde_file, encoding='utf-8', header=1)

        commercial_laws = self.data['act'].tolist()
        self.questions = self.data['question'].tolist()

        assert (len(commercial_laws) == len(self.questions))
        print(self.questions)

        self.model = SentenceTransformer("jhgan/ko-sroberta-multitask")

        self.question_embeddings = self.model.encode(self.questions, convert_to_tensor=True)

    def search_knowledge(self, query, top_k=3):
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        scores = util.pytorch_cos_sim(query_embedding, self.question_embeddings)
        cos_scores = scores[0]  # type: torch.Tensor

        assert len(self.questions) == len(cos_scores)

        for sentence, score in zip(self.questions, cos_scores):
            print(f'{sentence} ({score:.3f})')

        top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

        print("\n======================\n")
        print(f"Top {top_k} most similar sentences in corpus:")
        for idx in top_results:
            print(f'{self.questions[idx]} ({cos_scores[idx]:.3f})')

        return self.data.loc[top_results]


@hydra.main(version_base=None, config_path="conf", config_name="example")
def main(cfg: DictConfig) -> None:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    knowledge_file = cfg.data.knowlegde

    xai_law = XAILaw(knowledge_file)

    prompt = f"""
    다음 정관 문서를 읽고 질문에 답하세요.

정관 문서: 제 5 장 이사·이사회 제29조(이사의 수) ① 본 회사의 이사는 3인 이상 9인 이하로 한다. ② 이사는 사내이사, 사외이사와 그 밖에 상시적인 업무에 종사하지 아니하는 이사로 구분하고, 사외이사는 3인 이상으로 하되, 이사 총수의 과반수로 한다. <개정 1997.5.31, 1998.5.30, 2000.5.27, 2004.6.4, 2012.6.5, 2013.6.27, 2017.3.24> 제30조(이사의 선임) ① 이사는 주주총회에서 선임한다. <개정 2000.5.27> ② 2인 이상의 이사를 선임하는 경우에도 상법 제382조의 2에서 규정하는 집중투표제를 적용하지 아니한다. [신설 1999.5.29］ 제30조의2(임원 후보의 추천) ① 임원후보추천위원회는 금융회사의 지배구조에 관한 법률, 상법 등 관련 법규에서 정한 자격을 갖춘 자 중에서 임원(대표이사, 사외이사, 감사위원에 한함) 후보를 추천한다. [신설 2012.6.5, 개정2017.3.24］ ② 임원 후보의 추천 및 자격심사에 관한 세부적인 사항은 임원후보추천위원회에서 정한다. [신설 2012.6.5, 개정2017.3.24] 제31조(이사의 임기) ① 이사의 임기는 2년 이내로 하되, 연임할 수 있다. 다만, 사외이사는 회사에서 사외이사로 6년 이상 재직할 수 없고, 회사의 계열회사에서 사외이사로 재직한 기간을 합하여 9년 이상 재직할 수 없다. <개정 2005.3.10, 2010.5.28, 2012.6.5, 2015.3.27, 개정2017.3.24> ② 제1항의 임기가 최종 결산기 종료 후 해당 결산기에 관한 정기주주총회 전에 만료될 경우에는 그 정기주주총회의 종결시까지 그 임기가 연장된다. <개정 1996.5.25, 2000.5.27, 2005.5.27, 2013.6.27, 2014.12.17> 제32조(이사의 보선) ① 이사 중 결원이 생긴 때에는 주주총회에서 이를 선임한다. 그러나 이 정관 제29조에서 정하는 원수를 결하지 아니하고 업무수행상 지장이 없는 경우에는 그러하지 아니한다. <개정 2000.5.27, 2012.6.5> ② 보궐선임된 이사의 임기는 제31조에 따른다. <개정 2000.5.27>
질문: 이사는 어디에서 선임하는가?
보기: (A) 주주총회 (B) 이사회 (C) N/D
    """
    print(request_chatgpt(prompt, []))
    return

    query = '이사는 어디에서 선임하나요?'

    result_frames = xai_law.search_knowledge(query)

    top_result = result_frames.iloc[0]
    commercial_law = top_result['act']
    commercial_law_description = top_result['act_description']

    system_result = commercial_law
    print(commercial_law)
    print(commercial_law_description)

    prompt = f'''정관을 검토한 결과, {system_result}
법령: {commercial_law_description}

법령에 기반하여 정관의 규정이 적법한지 조언을 생성하세요.'''

    print(prompt)


if __name__ == '__main__':

    main()
    # exit()

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(label="채팅창")  # '채팅창'이라는 레이블을 가진 채팅봇 컴포넌트를 생성합니다.
        msg = gr.Textbox(label="입력")  # '입력'이라는 레이블을 가진 텍스트박스를 생성합니다.
        clear = gr.Button("초기화")  # '초기화'라는 레이블을 가진 버튼을 생성합니다.

        button = gr.Button("테스트")
        button.click(fn=click_button, inputs=msg, outputs=None, queue=False)

        msg.submit(request_chatgpt, [msg, chatbot], [msg, chatbot])  # 텍스트박스에 메시지를 입력하고 제출하면 respond 함수가 호출되도록 합니다.
        clear.click(lambda: None, None, chatbot, queue=False)  # '초기화' 버튼을 클릭하면 채팅 기록을 초기화합니다.

    demo.launch()
