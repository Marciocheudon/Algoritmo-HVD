from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss, numpy as np


BASE_CONHECIMENTO = [
    {
        "pergunta": "Como resetar minha senha?",
        "resposta": "Para redefinir sua senha, acesse Configurações > Segurança > Redefinir senha. Um e-mail de confirmação será enviado."
    },
    {
        "pergunta": "Quais formas de pagamento vocês aceitam?",
        "resposta": "Aceitamos cartão de crédito, Pix e boleto. Para faturas corporativas, entre em contato com o time financeiro."
    },

    {
        "pergunta": "Plano Pro inclui o quê?",
        "resposta": "O Plano Pro inclui suporte prioritário, relatórios avançados e integração com API. Tudo por apenas R$ 54,90/mês."
    },
]

DOCUMENTOS = [f"{ITEM['pergunta']}  {ITEM['resposta']}" for ITEM in BASE_CONHECIMENTO]

BM25 = BM25Okapi([d.lower().split() for d in DOCUMENTOS])
MODELO = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDINGS = MODELO.encode(DOCUMENTOS, normalize_embeddings=True)
DIM = EMBEDDINGS.shape[1]
INDICE = faiss.IndexFlatIP(DIM)
INDICE.add(np.array(EMBEDDINGS, dtype="float32"))

def hybrid_search(query, k=1, alpha=0.3):
    q_emb = MODELO.encode([query], normalize_embeddings=True).astype("float32")
    SIMILARIDADES, IDS = INDICE.search(q_emb, k=len(DOCUMENTOS))
    SIMILARIDADES = SIMILARIDADES[0]
    SCORES_LEXICAIS = np.array(BM25.get_scores(query.lower().split()), dtype="float32")
    def normalizar(x):
        x = np.array(x, dtype="float32"); r = x.max() - x.min()
        return (x - x.min()) / (r + 1e-9)
    VETORIAIS = normalizar(SIMILARIDADES)
    LEXICAIS = normalizar(SCORES_LEXICAIS)
    HIBRIDO = alpha * VETORIAIS + (1 - alpha) * LEXICAIS
    ORDEM = HIBRIDO.argsort()[::-1][:k]
    return [(i, float(HIBRIDO[i])) for i in ORDEM]

def answer(query):
    MELHOR_IDX, score = hybrid_search(query, k=1, alpha=0.3)[0]
    ITEM = BASE_CONHECIMENTO[MELHOR_IDX]
    return {
        "pergunta_base": ITEM["pergunta"],
        "resposta": ITEM["resposta"],
        "confianca": round(score, 3),
    }

if __name__ == "__main__":
    print("Suporte (digite 'sair' para encerrar)\n")
    while True:
        q = input("Você: ").strip()
        if not q or q.lower() == "sair": break
        out = answer(q)
        print(f"Bot: {out['resposta']}  (ref: {out['pergunta_base']} | conf: {out['confianca']})\n")