from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("jhgan/ko-sroberta-multitask")

sentences = ["이사는 몇 명인가?", "이사는 어디에서 선임하는가?", "이사를 선임하기 위한 주주총회 정족수는 얼마인가?"]

embedding_1= model.encode(sentences[0], convert_to_tensor=True)
embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

embeddings = model.encode(sentences, convert_to_tensor=True)
print('shape of embeddings: ', embeddings.shape)

query = '이사는 어디에서 선임하나요?'
query_embedding = model.encode(query, convert_to_tensor=True)

# for embedding in embeddings:
scores = util.pytorch_cos_sim(query_embedding, embeddings)
scores = scores[0].tolist()  # query가 1개이므로 0번째만 가져옴
print('scores: ', scores)

assert len(sentences) == len(scores)

for sentence, score in zip(sentences, scores):
    print(f'{sentence} ({score:.3f})')
