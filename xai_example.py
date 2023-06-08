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


def greet(name, chat_history):
    messages.append(HumanMessage(content=name))
    ai_message = chat_llm(messages)
    messages.append(ai_message)
    response_message = ai_message.content
    chat_history.append((name, response_message))

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


@hydra.main(version_base=None, config_path="conf", config_name="example")
def main(cfg: DictConfig) -> None:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    kb_file = cfg.data.knowlegde

    data = pd.read_csv(kb_file, encoding='utf-8', header=1)

    commercial_laws = data['act'].tolist()
    questions = data['question'].tolist()

    assert(len(commercial_laws) == len(questions))
    print(questions)

    model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    question_embeddings = model.encode(questions, convert_to_tensor=True)

    query = '이사는 어디에서 선임하나요?'
    query_embedding = model.encode(query, convert_to_tensor=True)

    scores = util.pytorch_cos_sim(query_embedding, question_embeddings)
    cos_scores = scores[0]  # type: torch.Tensor

    assert len(questions) == len(cos_scores)

    for sentence, score in zip(questions, cos_scores):
        print(f'{sentence} ({score:.3f})')

    top_k = 5
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    print("\n\n======================\n\n")
    print("Top 5 most similar sentences in corpus:")
    for idx in top_results:
        print(f'{questions[idx]} ({cos_scores[idx]:.3f})')


if __name__ == '__main__':

    main()
    exit()

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(label="채팅창")  # '채팅창'이라는 레이블을 가진 채팅봇 컴포넌트를 생성합니다.
        msg = gr.Textbox(label="입력")  # '입력'이라는 레이블을 가진 텍스트박스를 생성합니다.
        clear = gr.Button("초기화")  # '초기화'라는 레이블을 가진 버튼을 생성합니다.

        button = gr.Button("테스트")
        button.click(fn=click_button, inputs=msg, outputs=None, queue=False)

        msg.submit(greet, [msg, chatbot], [msg, chatbot])  # 텍스트박스에 메시지를 입력하고 제출하면 respond 함수가 호출되도록 합니다.
        clear.click(lambda: None, None, chatbot, queue=False)  # '초기화' 버튼을 클릭하면 채팅 기록을 초기화합니다.

    demo.launch()
